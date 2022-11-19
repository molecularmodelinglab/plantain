import torch
import torch.nn as nn
import torch.nn.functional as F

from dgllife.model.gnn import MPNNGNN
from dgllife.model import WeightedSumAndMax

from models.cat_scal_embedding import CatScalEmbedding

class InteractionGNN(nn.Module):

    def __init__(self, cfg, in_node):
        super().__init__()

        self.node_embed = CatScalEmbedding(cfg.model.node_embed_size,
                                           in_node.out_type_data.graph.ndata)
        self.edge_embed = CatScalEmbedding(cfg.model.edge_embed_size,
                                           in_node.out_type_data.graph.edata)

        self.gnn = MPNNGNN(self.node_embed.total_dim,
                           self.edge_embed.total_dim,
                           cfg.model.out_size,
                           cfg.model.edge_hidden_size,
                           cfg.model.num_mpnn_layers)

        self.readout = WeightedSumAndMax(cfg.model.out_size)

        combined_hid_sz = 2*cfg.model.out_size

        self.out_nns = nn.ModuleList()            
        for prev, size in zip([combined_hid_sz] + list(cfg.model.out_mlp_sizes), cfg.model.out_mlp_sizes):
            self.out_nns.append(nn.Sequential(
                nn.Linear(prev, size),
                nn.LayerNorm(size),
                nn.LeakyReLU(),
                nn.Dropout(cfg.model.dropout_rate),
            ))
        self.out = nn.Linear(cfg.model.out_mlp_sizes[-1], 1)

    def forward(self, batch):
        graph = batch.graph.dgl_batch
        node_feat = self.node_embed(batch.graph.ndata)
        edge_feat = self.edge_embed(batch.graph.edata)

        gnn_out = self.gnn(graph, node_feat, edge_feat)
        x = self.readout(graph, gnn_out)
        for nn in self.out_nns:
            x = nn(x)
        return self.out(x).squeeze(-1)