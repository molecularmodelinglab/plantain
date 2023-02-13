import torch
import torch.nn as nn
import torch.nn.functional as F
from common.pose_transform import Pose
from models.attention_gnn import MPNN
from terrace import Module
from terrace.batch import Batch
from terrace.dataframe import DFRow
from terrace.module import LazyLayerNorm, LazyLinear, LazyMultiheadAttention
from .model import ClassifyActivityModel
from .graph_embedding import GraphEmbedding

def rbf_encode(dists, start, end, steps):
    mu = torch.linspace(start, end, steps, device=dists.device)
    sigma = (start - end)/steps
    dists_expanded = dists.unsqueeze(-1).repeat(1,1,mu.size(0))
    mu_expanded = mu.view(1,1,-1)
    diff = ((dists_expanded - mu_expanded)/sigma)**2
    return torch.exp(-diff)

class ForceField(Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.model

    def get_hidden_feat(self, x):
        self.start_forward()

        lig_cfg = self.cfg.lig_encoder
        rec_cfg = self.cfg.rec_encoder

        lig_node_feats, lig_edge_feats = self.make(GraphEmbedding, self.cfg.lig_encoder)(x.lig_graph)
        rec_node_feats, rec_edge_feats = self.make(GraphEmbedding, self.cfg.rec_encoder)(x.rec_graph)

        lig_hid = self.make(LazyLinear, lig_cfg.node_out_size)(F.leaky_relu(lig_node_feats))
        rec_hid = self.make(LazyLinear, rec_cfg.node_out_size)(F.leaky_relu(rec_node_feats))

        if self.cfg.get("use_layer_norm", False):
            rec_hid = self.make(LazyLayerNorm)(rec_hid)
            lig_hid = self.make(LazyLayerNorm)(lig_hid)

        prev_lig_hid = []
        prev_rec_hid = []

        for layer in range(self.cfg.num_mpnn_layers):
            lig_hid = self.make(MPNN)(x.lig_graph, F.leaky_relu(lig_hid), lig_edge_feats)
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
                   batch_lig_poses):
        
        all_Us = []

        rec_graph = batch.rec_graph.dgl()
        lig_graph = batch.lig_graph.dgl()

        tot_rec = 0
        tot_lig = 0
        for i, (r, l) in enumerate(zip(rec_graph.batch_num_nodes(), lig_graph.batch_num_nodes())):
            
            lig_feat = batch_lig_feat[tot_lig:tot_lig+l]
            rec_feat = batch_rec_feat[tot_rec:tot_rec+r]

            rec_coord =  batch.rec_graph.ndata.coord[tot_rec:tot_rec+r]

            tot_rec += r
            tot_lig += l

            atn_coefs = torch.einsum('lef,ref->lre', lig_feat, rec_feat)

            # very hacky way of allowing diffusion model to pass multiple transforms
            # to the energy function
            if len(batch_lig_poses[i].coord.shape) == 3:
                Us = []
                for lig_coord in batch_lig_poses.coord[i]:
                    lc_ex = lig_coord.unsqueeze(1).expand(-1,rec_coord.size(0),-1)
                    rc_ex = rec_coord.unsqueeze(0).expand(lig_coord.size(0),-1,-1)
                    dists = torch.sqrt(((lc_ex - rc_ex)**2).sum(-1))
                    rbfs = rbf_encode(dists, self.cfg.rbf_start,self.cfg.rbf_end,self.cfg.rbf_steps)

                    U = (atn_coefs*rbfs*self.cfg.energy_scale).sum()
                    Us.append(U)
                all_Us.append(torch.stack(Us))
            else:
                assert len(batch_lig_poses[i].coord.shape) == 2
                lig_coord =  batch_lig_poses[i].coord

                lc_ex = lig_coord.unsqueeze(1).expand(-1,rec_coord.size(0),-1)
                rc_ex = rec_coord.unsqueeze(0).expand(lig_coord.size(0),-1,-1)
                dists = torch.sqrt(((lc_ex - rc_ex)**2).sum(-1))
                rbfs = rbf_encode(dists, self.cfg.rbf_start,self.cfg.rbf_end,self.cfg.rbf_steps)

                U = (atn_coefs*rbfs*self.cfg.energy_scale).sum()
                all_Us.append(U)

        return torch.stack(all_Us)

class ForceFieldClassifier(Module, ClassifyActivityModel):
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.model
        self.force_field = ForceField(cfg)

    @staticmethod
    def get_name() -> str:
        return "force_field"

    def get_input_feats(self):
        return ["lig_graph", "rec_graph", "lig_docked_poses"]

    def forward(self, batch):
        self.start_forward()

        batch_rec_feat, batch_lig_feat = self.force_field.get_hidden_feat(batch)
        ret = []
        for conf_id in range(len(batch.lig_docked_poses.coord[0])):
            pose = Batch(Pose, coord=[coord[conf_id] for coord in batch.lig_docked_poses.coord])
            ret.append(self.force_field.get_energy(batch, batch_rec_feat, batch_lig_feat, pose))
        U = torch.stack(ret).T

        if self.cfg.get("multi_pose_attention", False):
            lin_out = self.make(LazyLinear, 2)(U.unsqueeze(-1))
            pose_scores = lin_out[...,0]
            atn = lin_out[...,1]
            atn = torch.softmax(atn[:,:-1], -1)
            score = (pose_scores[:,:-1]*atn).sum(-1)
        else:
            score = U[:,0]
            pose_scores = U

        return Batch(DFRow, score=score, pose_scores=pose_scores)
