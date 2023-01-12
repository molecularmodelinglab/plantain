import torch
import torch.nn.functional as F
from data_formats.graphs.graph_formats import LigAndRecGraph
from terrace import Module, LazyLinear
from dgl.nn.pytorch import NNConv
from data_formats.tasks import ClassifyActivity
from .model import ClassifyActivityModel
from .graph_embedding import GraphEmbedding

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

def batched_outer_prod(x, batch_lig_feat, batch_rec_feat):
    tot_rec = 0
    tot_lig = 0
    ret = []
    for l, r in zip(x.lig_graph.dgl().batch_num_nodes(), x.rec_graph.dgl().batch_num_nodes()):
        lig_feat = batch_lig_feat[tot_lig:tot_lig+l]
        rec_feat = batch_rec_feat[tot_rec:tot_rec+r]

        tot_rec += r
        tot_lig += l

        op = torch.einsum('lf,rf->lr', lig_feat, rec_feat)
        ret.append(op)

    return ret


class AttentionGNN(Module, ClassifyActivityModel):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.model

    @staticmethod
    def get_name():
        return "attention_gnn"

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

        prev_lig_hid = []
        prev_rec_hid = []

        for layer in range(self.cfg.num_mpnn_layers):
            lig_hid = self.make(MPNN)(x.lig_graph, F.leaky_relu(lig_hid), lig_edge_feats)
            rec_hid = self.make(MPNN)(x.rec_graph, F.leaky_relu(rec_hid), rec_edge_feats)

            # make it residual!
            prev_layer = layer - 2
            if prev_layer >= 0:
                lig_hid = lig_hid + prev_lig_hid[prev_layer]
                rec_hid = rec_hid + prev_rec_hid[prev_layer]

            prev_lig_hid.append(lig_hid)
            prev_rec_hid.append(rec_hid)


        ops = batched_outer_prod(x, lig_hid, rec_hid)
        out = torch.stack([ op.mean() for op in ops ])

        return out
            