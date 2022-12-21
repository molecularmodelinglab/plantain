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
                                           in_node.out_type_data.graphs[0].ndata)
        self.edge_embed = CatScalEmbedding(cfg.model.edge_embed_size,
                                           in_node.out_type_data.graphs[0].edata)

        self.gnn = MPNNGNN(self.node_embed.total_dim,
                           self.edge_embed.total_dim,
                           cfg.model.node_hidden_size,
                           cfg.model.edge_hidden_size,
                           cfg.model.num_mpnn_layers)

        self.node_out = nn.Linear(cfg.model.node_hidden_size, cfg.model.out_size)

        self.readout = WeightedSumAndMax(cfg.model.out_size)

        combined_hid_sz = 2*cfg.model.out_size

        self.weight_nn = nn.Linear(combined_hid_sz, 1)
        self.out_nns = nn.ModuleList()            
        for prev, size in zip([combined_hid_sz] + list(cfg.model.out_mlp_sizes), cfg.model.out_mlp_sizes):
            self.out_nns.append(nn.Sequential(
                nn.Linear(prev, size),
                nn.LayerNorm(size),
                nn.LeakyReLU(),
                nn.Dropout(cfg.model.dropout_rate),
            ))
        self.out = nn.Linear(cfg.model.out_mlp_sizes[-1], 1)

    def get_conformer_scores(self, batch):
        """ Return the unnormalized beta values (i.e. the scores 
        of each conformers) """
        xs = []
        for graph in batch.graphs:
            node_feat = self.node_embed(graph.ndata)
            edge_feat = self.edge_embed(graph.edata)

            gnn_out = self.gnn(graph.dgl_batch, node_feat, edge_feat)
            node_out = self.node_out(gnn_out)
            x = self.readout(graph.dgl_batch, node_out)
            xs.append(x)

        x = torch.stack(xs, 1) #(B, C, F)
        # attention-like mechanism along conformer dim
        beta_unnorm = self.weight_nn(x)

        return beta_unnorm

    def forward(self, batch):
        xs = []
        for graph in batch.graphs:
            node_feat = self.node_embed(graph.ndata)
            edge_feat = self.edge_embed(graph.edata)

            gnn_out = self.gnn(graph.dgl_batch, node_feat, edge_feat)
            node_out = self.node_out(gnn_out)
            x = self.readout(graph.dgl_batch, node_out)
            xs.append(x)

        x = torch.stack(xs, 1) #(B, C, F)
        # attention-like mechanism along conformer dim
        beta_unnorm = self.weight_nn(x)
        beta = torch.softmax(beta_unnorm, 1)
        x = torch.sum(x*beta, 1)

        for nn in self.out_nns:
            x = nn(x)
        return self.out(x).squeeze(-1)