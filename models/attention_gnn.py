import torch
import torch.nn as nn
import torch.nn.functional as F
from data_formats.base_formats import Data, InvDistMat, Prediction
from data_formats.graphs.graph_formats import LigAndRecGraph
from terrace import Module, LazyLinear, LazyMultiheadAttention, LazyLayerNorm, Batch
from dgl.nn.pytorch import NNConv
from data_formats.tasks import ClassifyActivity, PredictInteractionMat, RejectOption, ScoreActivityClass
from .model import ClassifyActivityModel
from .graph_embedding import GraphEmbedding

class PredBCE(Prediction):
    pred_bce: float

class PredDocked(Prediction):
    docked_score: float

# class Select(Prediction):
#     select_unnorm: float
#     select_prob: float

class MPNN(Module):
    """ Very basic message passing step"""

    def __init__(self):
        super().__init__()

    def forward(self, g, node_feats, edge_feats):
        self.start_forward()
        node_size = node_feats.shape[-1]
        edge_func = self.make(LazyLinear, node_size*node_size)
        gnn = self.make(NNConv, node_size, node_size, edge_func, 'mean')
        x = gnn(g.dgl(), node_feats, edge_feats)
        return self.make(LazyLinear, node_size)(F.leaky_relu(x))

def batched_feats(x, batch_lig_feat, batch_rec_feat):
    tot_rec = 0
    tot_lig = 0
    ret = []
    for l, r in zip(x.lig_graph.dgl().batch_num_nodes(), x.rec_graph.dgl().batch_num_nodes()):
        lig_feat = batch_lig_feat[tot_lig:tot_lig+l]
        rec_feat = batch_rec_feat[tot_rec:tot_rec+r]

        tot_rec += r
        tot_lig += l

        yield lig_feat, rec_feat
        # op = torch.einsum('lf,rf->lr', lig_feat, rec_feat)
        # ret.append(op)

class AttentionGNN(Module, ClassifyActivityModel):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.model
        if self.cfg.get("use_slope_intercept", False):
            self.slope = nn.Parameter(torch.tensor(0.0))
            self.intercept = nn.Parameter(torch.tensor(0.1))

    @staticmethod
    def get_name():
        return "attention_gnn"

    def get_tasks(self):
        return [ ScoreActivityClass, ClassifyActivity, PredictInteractionMat, RejectOption ]

    def get_data_format(self):
        return LigAndRecGraph.make

    def forward(self, x):

        self.start_forward()

        lig_cfg = self.cfg.lig_encoder
        rec_cfg = self.cfg.rec_encoder

        lig_node_feats, lig_edge_feats = self.make(GraphEmbedding, self.cfg.lig_encoder)(x.lig_graph)
        rec_node_feats, rec_edge_feats = self.make(GraphEmbedding, self.cfg.rec_encoder)(x.rec_graph)

        lig_hid = self.make(LazyLinear, lig_cfg.node_out_size)(F.leaky_relu(lig_node_feats))
        rec_hid = self.make(LazyLinear, rec_cfg.node_out_size)(F.leaky_relu(rec_node_feats))

        # lig_hid = lig_hid.type(torch.complex64)
        # rec_hid = rec_hid.type(torch.complex64)

        if self.cfg.get("use_layer_norm", False):
            # lig_hid = torch.view_as_real(lig_hid)
            # rec_hid = torch.view_as_real(rec_hid)
            rec_hid = self.make(LazyLayerNorm)(rec_hid)
            lig_hid = self.make(LazyLayerNorm)(lig_hid)
            # lig_hid = torch.view_as_complex(lig_hid)
            # rec_hid = torch.view_as_complex(rec_hid)

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

        # add null rec atom to provide option for ligands
        # to not interact with anything
        if self.cfg.get("use_null_residue", False):
            null_data = torch.rand((1, rec_hid.size(-1)))
            null = self.make_param(nn.Parameter, null_data)

        ops = []
        for lig_feat, rec_feat in batched_feats(x, lig_hid, rec_hid):
            if self.cfg.get("use_null_residue", False):
                rec_feat = torch.cat((rec_feat, null), 0)
            op = torch.einsum('lf,rf->lr', lig_feat, rec_feat)
            ops.append(op)

        if self.cfg.get("use_slope_intercept", False):
            ops = [ op*self.slope + self.intercept for op in ops ]

        if self.cfg.get("use_null_residue", False):
            # take out the null value when returning the activity
            out_inp = torch.stack([ op[:, :-1].mean() for op in ops ]).unsqueeze(-1)
            null_inp = torch.stack([ op[:, -1:].mean() for op in ops ]).unsqueeze(-1)

            out_inp = torch.cat((out_inp, null_inp), -1)
            out = self.make(LazyLinear, 1)(out_inp)[:,0]
        else:
            out = torch.stack([ op.mean() for op in ops ])

        if self.cfg.get("predict_bce", False):
            rec_bce_hid = self.make(LazyLinear, rec_hid.shape[-1])(rec_hid)
            lig_bce_hid = self.make(LazyLinear, lig_hid.shape[-1])(lig_hid)
            
            bce_ops = []
            for lig_feat, rec_feat in batched_feats(x, lig_bce_hid, rec_bce_hid):
                if self.cfg.get("use_null_residue", False):
                    rec_feat = torch.cat((rec_feat, null), 0)
                op = torch.einsum('lf,rf->lr', lig_feat, rec_feat)
                bce_ops.append(op)

            bce_out = torch.stack([ op[:, :-1].mean() for op in bce_ops ])
        else:
            bce_out = None

        if self.cfg.get("predict_select", False):
            rec_bce_hid = self.make(LazyLinear, rec_hid.shape[-1])(rec_hid)
            lig_bce_hid = self.make(LazyLinear, lig_hid.shape[-1])(lig_hid)
            
            bce_ops = []
            for lig_feat, rec_feat in batched_feats(x, lig_bce_hid, rec_bce_hid):
                if self.cfg.get("use_null_residue", False):
                    rec_feat = torch.cat((rec_feat, null), 0)
                op = torch.einsum('lf,rf->lr', lig_feat, rec_feat)
                bce_ops.append(op)

            select_unnorm = torch.stack([ op[:, :-1].mean() for op in bce_ops ])
        else:
            select_unnorm = None

        if self.cfg.get("predict_docked_score", False):
            rec_bce_hid = self.make(LazyLinear, rec_hid.shape[-1])(rec_hid)
            lig_bce_hid = self.make(LazyLinear, lig_hid.shape[-1])(lig_hid)
            
            bce_ops = []
            for lig_feat, rec_feat in batched_feats(x, lig_bce_hid, rec_bce_hid):
                if self.cfg.get("use_null_residue", False):
                    rec_feat = torch.cat((rec_feat, null), 0)
                op = torch.einsum('lf,rf->lr', lig_feat, rec_feat)
                bce_ops.append(op)

            docked_score = torch.stack([ op[:, :-1].mean() for op in bce_ops ])
        else:
            docked_score = None

        return out, ops, bce_out, select_unnorm, docked_score

    def predict(self, tasks, x):
        score, mats, bce_pred, select_unnorm, docked_score = self(x)
        p1 = super().make_prediction(score)
        p2 = Batch(InvDistMat, inv_dist_mat=mats)
        ret = [ p1, p2 ]
        if bce_pred is not None:
            ret.append(Batch(PredBCE, pred_bce=bce_pred))
            if select_unnorm is None:
                ret.append(Batch(RejectOption.Prediction, select_score=-bce_pred))
        if select_unnorm is not None:
            ret.append(Batch(RejectOption.Prediction, select_score=select_unnorm))
        
        # very hacky way to get gnina disagreement
        if select_unnorm is None and bce_pred is None and hasattr(x, "pose_scores"):
            pose_bce = docked_pose_bce(x, p2)
            ret.append(Batch(RejectOption.Prediction, select_score=pose_bce))

        if docked_score is not None:
            ret.append(Batch(PredDocked, docked_score=docked_score))
        
        return Data.merge(ret)

def docked_pose_bce(x, pred):
    bces = []
    for x0, pred0 in zip(x, pred):
        rec_coord = x0.rec_graph.ndata.coord
        lig_coord = x0.lig_graph.ndata.coord
        dist = torch.cdist(lig_coord, rec_coord)

        nulls = torch.tensor([[5.0]]*len(lig_coord), device=lig_coord.device)
        dist = torch.cat((dist, nulls), 1)
        labels = torch.argmin(dist, 1)

        bces.append(F.cross_entropy(pred0.inv_dist_mat, labels))
    return torch.stack(bces)