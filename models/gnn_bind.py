import torch
import torch.nn as nn
import torch.nn.functional as F

from dgllife.model.gnn import MPNNGNN
from dgllife.model import WeightedSumAndMax

def get_gnns(init_node_sz,
             hid_node_sizes,
             init_edge_sz,
             hid_edge_sz,
             num_mpnn_layers):
    node_sizes = [ init_node_sz ] + list(hid_node_sizes)
    ret = nn.ModuleList()
    for ps, cs in zip(node_sizes, node_sizes[1:]):
        ret.append(MPNNGNN(ps, init_edge_sz, cs, hid_edge_sz, num_mpnn_layers))
    return ret

class CatScalEmbedding(nn.Module):

    def __init__(self, embed_size, td):
        super().__init__()
        self.embeddings = nn.ModuleList()
        total_dim = td.scal_feat.shape[-1]
        for i, val in enumerate(td.cat_feat.max_values):
            embedding = nn.Embedding(val, embed_size)
            total_dim += embed_size
            self.embeddings.append(embedding)
        self.total_dim = total_dim

    def forward(self, batch):
        ret = [ batch.scal_feat ]
        for i, embed in enumerate(self.embeddings):
            ret.append(embed(batch.cat_feat[:,i]))
        return torch.cat(ret, -1)

class GNNBind(nn.Module):
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

        self.lig_gnns = get_gnns(self.lig_node_embed.total_dim,
                                 lig_cfg.node_hidden_sizes,
                                 self.lig_edge_embed.total_dim,
                                 lig_cfg.edge_hidden_size,
                                 lig_cfg.num_mpnn_layers)

        self.rec_gnns = get_gnns(self.rec_node_embed.total_dim,
                                 rec_cfg.node_hidden_sizes,
                                 self.rec_edge_embed.total_dim,
                                 rec_cfg.edge_hidden_size,
                                 rec_cfg.num_mpnn_layers)
                                 
        self.attentions = nn.ModuleList()
        self.lig_combiners = nn.ModuleList()
        for lig_sz, rec_sz in zip(lig_cfg.node_hidden_sizes, rec_cfg.node_hidden_sizes):
            # todo: this doesn't work when lig_size != rec_sz
            self.attentions.append(nn.MultiheadAttention(lig_sz, 1, kdim=rec_sz, vdim=rec_sz))
            comb_sz = lig_sz*2 #+rec_sz
            self.lig_combiners.append(nn.Sequential(
                nn.Linear(comb_sz, lig_sz),
                nn.LayerNorm(lig_sz),
                nn.LeakyReLU()
            ))

        assert len(self.lig_gnns) == len(self.rec_gnns)

        self.rec_readout = WeightedSumAndMax(rec_cfg.node_hidden_sizes[-1])
        self.lig_readout = WeightedSumAndMax(lig_cfg.node_hidden_sizes[-1])
        self.out_nns = nn.ModuleList()
        combined_hid_sz = 2*(rec_cfg.node_hidden_sizes[-1] + lig_cfg.node_hidden_sizes[-1])
            
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
        for lig_gnn, rec_gnn, atn, comb in zip(self.lig_gnns, self.rec_gnns, self.attentions, self.lig_combiners):
            rec_hid = rec_gnn(rec_graph, rec_hid, rec_edge_feat)
            lig_hid = lig_gnn(lig_graph, lig_hid, lig_edge_feat)

            tot_rec = 0
            tot_lig = 0
            atn_list = []
            for r, l in zip(rec_graph.batch_num_nodes(), lig_graph.batch_num_nodes()):
                query = lig_hid[tot_lig:tot_lig+l]
                key_val = rec_hid[tot_rec:tot_rec+r]
                tot_rec += r
                tot_lig += l
                lig_rec_atn, _ = atn(query, key_val, key_val)
                atn_list.append(lig_rec_atn)
            comb_atn = torch.cat(atn_list, 0)
            lig_hid = comb(torch.cat((lig_hid, comb_atn), axis=-1))

        rec_readout = self.rec_readout(rec_graph, rec_hid)
        lig_readout = self.lig_readout(lig_graph, lig_hid)
        x = torch.cat((rec_readout, lig_readout), -1)
        for nn in self.out_nns:
            x = nn(x)
        return self.out(x).squeeze(-1)