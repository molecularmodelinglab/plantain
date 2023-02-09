import torch
import torch.nn as nn
import torch.nn.functional as F
from data_formats.base_formats import Data, Prediction
from data_formats.graphs.graph_formats import LigAndRecGraphMultiPose
from models.attention_gnn import MPNN
from terrace import Module
from terrace.batch import Batch
from terrace.module import LazyLayerNorm, LazyLinear, LazyMultiheadAttention
from .model import ClassifyActivityModel
from .graph_embedding import GraphEmbedding

class PoseScores(Prediction):
    pose_scores: torch.Tensor

def rbf_encode(dists, start, end, steps):
    mu = torch.linspace(start, end, steps, device=dists.device)
    sigma = (start - end)/steps
    dists_expanded = dists.unsqueeze(-1).repeat(1,1,mu.size(0))
    mu_expanded = mu.view(1,1,-1)
    diff = ((dists_expanded - mu_expanded)/sigma)**2
    return torch.exp(-diff)

class ForceField(Module, ClassifyActivityModel):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.model

    @staticmethod
    def get_name() -> str:
        return "force_field"

    def get_data_format(self):
        return LigAndRecGraphMultiPose.make

    def get_hidden_feat(self, x, conf_id):

        lig_cfg = self.cfg.lig_encoder
        rec_cfg = self.cfg.rec_encoder

        lig_node_feats, lig_edge_feats = self.make(GraphEmbedding, self.cfg.lig_encoder)(x.lig_graphs[conf_id])
        rec_node_feats, rec_edge_feats = self.make(GraphEmbedding, self.cfg.rec_encoder)(x.rec_graph)

        lig_hid = self.make(LazyLinear, lig_cfg.node_out_size)(F.leaky_relu(lig_node_feats))
        rec_hid = self.make(LazyLinear, rec_cfg.node_out_size)(F.leaky_relu(rec_node_feats))

        if self.cfg.get("use_layer_norm", False):
            rec_hid = self.make(LazyLayerNorm)(rec_hid)
            lig_hid = self.make(LazyLayerNorm)(lig_hid)

        prev_lig_hid = []
        prev_rec_hid = []

        for layer in range(self.cfg.num_mpnn_layers):
            lig_hid = self.make(MPNN)(x.lig_graphs[conf_id], F.leaky_relu(lig_hid), lig_edge_feats)
            rec_hid = self.make(MPNN)(x.rec_graph, F.leaky_relu(rec_hid), rec_edge_feats)

            if self.cfg.get("use_layer_norm", False):
                rec_hid = self.make(LazyLayerNorm)(rec_hid)
                lig_hid = self.make(LazyLayerNorm)(lig_hid)

            if self.cfg.get("inner_attention", False):
                rec_hid = (rec_hid + self.make(LazyMultiheadAttention, 1)(rec_hid, lig_hid, lig_hid)[0])
                lig_hid = (lig_hid + self.make(LazyMultiheadAttention, 1)(lig_hid, rec_hid, rec_hid)[0])
                if self.cfg.get("use_layer_norm", False):
                    rec_hid = self.make(LazyLayerNorm)(rec_hid)
                    lig_hid = self.make(LazyLayerNorm)(lig_hid)

            # make it residual!
            prev_layer = layer - 2
            if prev_layer >= 0:
                lig_hid = lig_hid + prev_lig_hid[prev_layer]
                rec_hid = rec_hid + prev_rec_hid[prev_layer]

            prev_lig_hid.append(lig_hid)
            prev_rec_hid.append(rec_hid)

        lig_hid = self.make(LazyLinear, self.cfg.out_size*self.cfg.rbf_steps)(F.leaky_relu(lig_hid))
        rec_hid = self.make(LazyLinear, self.cfg.out_size*self.cfg.rbf_steps)(F.leaky_relu(rec_hid))

        if self.cfg.get("use_layer_norm", False):
            rec_hid = self.make(LazyLayerNorm)(rec_hid)
            lig_hid = self.make(LazyLayerNorm)(lig_hid)

        lig_hid = lig_hid.view(-1, self.cfg.rbf_steps, self.cfg.out_size)
        rec_hid = rec_hid.view(-1, self.cfg.rbf_steps, self.cfg.out_size)

        return rec_hid, lig_hid
        
    def get_energy(self,
                   batch,
                   batch_rec_feat,
                   batch_lig_feat,
                   conf_id):
        
        all_Us = []

        rec_graph = batch.rec_graph.dgl()
        lig_graph = batch.lig_graphs[conf_id].dgl()

        tot_rec = 0
        tot_lig = 0
        for i, (r, l) in enumerate(zip(rec_graph.batch_num_nodes(), lig_graph.batch_num_nodes())):
            
            lig_feat = batch_lig_feat[tot_lig:tot_lig+l]
            rec_feat = batch_rec_feat[tot_rec:tot_rec+r]

            lig_coord =  batch.lig_graphs[conf_id].ndata.coord[tot_lig:tot_lig+l]
            rec_coord =  batch.rec_graph.ndata.coord[tot_rec:tot_rec+r]

            tot_rec += r
            tot_lig += l

            atn_coefs = torch.einsum('lef,ref->lre', lig_feat, rec_feat)

            # cdist is the simplest way to compute dist matrix
            # dists = torch.cdist(new_lig_coord, rec_coord)
            
            # non-cdist vectorized way
            lc_ex = lig_coord.unsqueeze(1).expand(-1,rec_coord.size(0),-1)
            rc_ex = rec_coord.unsqueeze(0).expand(lig_coord.size(0),-1,-1)
            dists = torch.sqrt(((lc_ex - rc_ex)**2).sum(-1))
            rbfs = rbf_encode(dists, self.cfg.rbf_start,self.cfg.rbf_end,self.cfg.rbf_steps)

            U = (atn_coefs*rbfs*self.cfg.energy_scale).sum()
            all_Us.append(U)

        return torch.stack(all_Us)

    def forward(self, batch):
        self.start_forward()

        batch_rec_feat, batch_lig_feat = self.get_hidden_feat(batch, 0)
        ret = []
        for conf_id in range(len(batch.lig_graphs)):
            ret.append(self.get_energy(batch, batch_rec_feat, batch_lig_feat, conf_id))
        U = torch.stack(ret).T

        if self.cfg.get("multi_pose_attention", False):
            lin_out = self.make(LazyLinear, 2)(U.unsqueeze(-1))
            pose_scores = lin_out[...,0]
            atn = lin_out[...,1]
            atn = torch.softmax(atn[:,:-1], -1)
            score = (pose_scores[:,:-1]*atn).sum(-1)
        else:
            score = U[:,0]

        return score, pose_scores

    def predict(self, tasks, x):
        score, pose_scores = self(x)
        p1 = super().make_prediction(score)
        p2 = Batch(PoseScores, pose_scores=pose_scores)
        return Data.merge((p1,p2))
