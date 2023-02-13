import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit.Chem import Descriptors
from terrace import Module, LazyLinear, LazyMultiheadAttention, LazyLayerNorm, Batch
from dgl.nn.pytorch import NNConv, WeightAndSum
from .model import ClassifyActivityModel
from .attention_gnn import MPNN
from .graph_embedding import GraphEmbedding

# class LigAndRecDescriptors(Input):
#     rec_diam: float
#     lig_clogp: float
#     clogp_x_diam: float
    
#     @staticmethod
#     def make(cfg, data: LigAndRecGraph):
#         rec_diam = torch.cdist(data.lig_graph.ndata.coord, data.rec_graph.ndata.coord).max()
#         lig_clogp = torch.tensor(Descriptors.MolLogP(data.lig), dtype=torch.float32)
#         clogp_x_diam = rec_diam*lig_clogp
#         return LigAndRecDescriptors(rec_diam, lig_clogp, clogp_x_diam)

#     @staticmethod
#     def make_full(cfg, data: LigAndRec):
#         graph_data = LigAndRecGraph.make(cfg, data)
#         arg = Data.merge([data, graph_data])
#         return Data.merge([graph_data, LigAndRecDescriptors.make(cfg, arg)])

class AttentionGNNNew(Module, ClassifyActivityModel):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.model

    @staticmethod
    def get_name():
        return "attention_gnn_new"

    def get_tasks(self):
        return [ ScoreActivityClass, ClassifyActivity, PredictInteractionMat, RejectOption ]

    def get_data_format(self):
        return LigAndRecDescriptors.make_full

    def forward(self, x):

        self.start_forward()

        lig_cfg = self.cfg.lig_encoder
        rec_cfg = self.cfg.rec_encoder

        lig_node_feats, lig_edge_feats = self.make(GraphEmbedding, self.cfg.lig_encoder)(x.lig_graph)
        rec_node_feats, rec_edge_feats = self.make(GraphEmbedding, self.cfg.rec_encoder)(x.rec_graph)

        lig_hid = self.make(LazyLinear, lig_cfg.node_out_size)(F.leaky_relu(lig_node_feats))
        rec_hid = self.make(LazyLinear, rec_cfg.node_out_size)(F.leaky_relu(rec_node_feats))
        rec_hid = self.make(LazyLayerNorm)(rec_hid)
        lig_hid = self.make(LazyLayerNorm)(lig_hid)

        prev_lig_hid = []
        prev_rec_hid = []

        for layer in range(self.cfg.num_mpnn_layers):
            lig_hid = self.make(MPNN)(x.lig_graph, F.leaky_relu(lig_hid), lig_edge_feats)
            rec_hid = self.make(MPNN)(x.rec_graph, F.leaky_relu(rec_hid), rec_edge_feats)
            rec_hid = self.make(LazyLayerNorm)(rec_hid)
            lig_hid = self.make(LazyLayerNorm)(lig_hid)

            rec_lig_hid = self.make(LazyMultiheadAttention, 1)(rec_hid, lig_hid, lig_hid)[0]
            lig_rec_hid = self.make(LazyMultiheadAttention, 1)(lig_hid, rec_hid, rec_hid)[0]

            rec_hid = self.make(LazyLinear, rec_hid.size(-1))(torch.cat((rec_hid, rec_lig_hid), -1))
            lig_hid = self.make(LazyLinear, lig_hid.size(-1))(torch.cat((lig_hid, lig_rec_hid), -1))

            rec_hid = self.make(LazyLayerNorm)(rec_hid)
            lig_hid = self.make(LazyLayerNorm)(lig_hid)

            # make it residual!
            prev_layer = layer - 2
            if prev_layer >= 0:
                lig_hid = lig_hid + prev_lig_hid[prev_layer]
                rec_hid = rec_hid + prev_rec_hid[prev_layer]

            prev_lig_hid.append(lig_hid)
            prev_rec_hid.append(rec_hid)

        lig_feats = self.make(WeightAndSum, lig_hid.size(-1))(x.lig_graph.dgl(), lig_hid)
        rec_feats = self.make(WeightAndSum, rec_hid.size(-1))(x.rec_graph.dgl(), rec_hid)
        if self.cfg.get("use_outer_prod", False):
            B = len(x)
            feats = torch.einsum('bi,bj->bij', lig_feats, rec_feats).view((B, -1))
        elif self.cfg.get("use_rec_out", False):
            feats = torch.cat([lig_feats, rec_feats], -1)
        else:
            feats = lig_feats

        for size in self.cfg.out_sizes:
            feats = self.make(LazyLinear, size)(F.leaky_relu(feats))
            feats = self.make(LazyLayerNorm)(feats)

        out = self.make(LazyLinear, 1)(F.leaky_relu(feats))[:,0]
        descriptors = {}
        for key in LigAndRecDescriptors.__annotations__.keys():
            descriptors[key] = self.make(LazyLinear, 1)(F.leaky_relu(feats))[:,0]

        return out, descriptors

    def predict(self, tasks, x):
        score, descriptors = self(x)
        p1 = super().make_prediction(score)
        p2 = Batch(LigAndRecDescriptors, **descriptors)
        # mse = F.mse_loss(x.clogp_x_diam, p2.clogp_x_diam, reduction='none')
        # score = -torch.tanh(torch.sqrt(mse)*0.005)
        logp_mse = F.mse_loss(x.lig_clogp, p2.lig_clogp, reduction='none')
        score = -torch.sqrt(logp_mse)
        p3 = Batch(RejectOption.Prediction, select_score = score)
        ret = [ p1, p2, p3]
        return Data.merge(ret)