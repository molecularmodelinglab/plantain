import torch
import torch.nn as nn
import torch.nn.functional as F

from dgllife.model.gnn import MPNNGNN
from dgllife.model import WeightedSumAndMax

from terrace.batch import Batch
from models.cat_scal_embedding import CatScalEmbedding
from datasets.data_types import PredData

class OuterProdGNN(nn.Module):
    def __init__(self, cfg, in_node):
        super().__init__()
        self.cfg = cfg

        rec_cfg = cfg.model.rec_encoder
        lig_cfg = cfg.model.lig_encoder

        self.rec_node_embed = CatScalEmbedding(cfg.model.rec_encoder.node_embed_size,
                                            in_node.out_type_data.rec.ndata)

        self.lig_node_embed = CatScalEmbedding(cfg.model.lig_encoder.node_embed_size,
                                            in_node.out_type_data.lig.ndata)

        self.rec_edge_embed = CatScalEmbedding(cfg.model.rec_encoder.edge_embed_size,
                                            in_node.out_type_data.rec.edata)

        self.lig_edge_embed = CatScalEmbedding(cfg.model.lig_encoder.edge_embed_size,
                                            in_node.out_type_data.lig.edata)

        self.lig_gnn = MPNNGNN(self.lig_node_embed.total_dim,
                               self.lig_edge_embed.total_dim,
                               lig_cfg.out_size,
                               lig_cfg.edge_hidden_size,
                               lig_cfg.num_mpnn_layers)

        self.rec_gnn = MPNNGNN(self.rec_node_embed.total_dim,
                               self.rec_edge_embed.total_dim,
                               rec_cfg.out_size,
                               rec_cfg.edge_hidden_size,
                               rec_cfg.num_mpnn_layers)

        self.rec_readout = WeightedSumAndMax(rec_cfg.out_size)
        self.lig_readout = WeightedSumAndMax(lig_cfg.out_size)

        combined_hid_sz = 4*(rec_cfg.out_size*lig_cfg.out_size)

        self.out_nns = nn.ModuleList()            
        for prev, size in zip([combined_hid_sz] + list(cfg.model.out_mlp_sizes), cfg.model.out_mlp_sizes):
            self.out_nns.append(nn.Sequential(
                nn.Linear(prev, size),
                nn.LayerNorm(size),
                nn.LeakyReLU()
            ))
        self.out = nn.Linear(cfg.model.out_mlp_sizes[-1], 1)
        
    def forward(self, batch):
        rec_graph = batch.rec.dgl_batch
        lig_graph = batch.lig.dgl_batch
        rec_hid = self.rec_node_embed(batch.rec.ndata)
        lig_hid = self.lig_node_embed(batch.lig.ndata)
        rec_edge_feat = self.rec_edge_embed(batch.rec.edata)
        lig_edge_feat = self.lig_edge_embed(batch.lig.edata)

        rec_out = self.rec_gnn(rec_graph, rec_hid, rec_edge_feat)
        lig_out = self.lig_gnn(lig_graph, lig_hid, lig_edge_feat)

        rec_readout = self.rec_readout(rec_graph, rec_out)
        lig_readout = self.lig_readout(lig_graph, lig_out)

        B = len(batch)
        x = torch.einsum('bi,bj->bij', rec_readout, lig_readout).view((B, -1))
        for nn in self.out_nns:
            x = nn(x)
        return self.out(x).squeeze(-1)