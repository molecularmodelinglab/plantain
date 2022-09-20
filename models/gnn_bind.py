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
    
class GNNBindModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # need to yeet this cat/scal thing. Holdover from equibind
        rec_cat_feats, rec_scal_feats = get_rec_node_feat_dims(cfg)
        lig_cat_feats, lig_scal_feats = get_lig_node_feat_dims(cfg)
        lig_node_feats = len(lig_cat_feats) + lig_scal_feats
        rec_edge_feats = sum(get_rec_edge_feat_dims(cfg))
        lig_edge_feats = sum(get_lig_edge_feat_dims(cfg))

        assert rec_scal_feats == 0
        assert len(rec_cat_feats) == 1
        
        rec_cfg = cfg.model.rec_encoder
        lig_cfg = cfg.model.lig_encoder
             
        self.rec_embed = nn.Embedding(rec_cat_feats[0], rec_cfg.embed_size)

        self.lig_gnns = get_gnns(lig_node_feats,
                                 lig_cfg.node_hidden_sizes,
                                 lig_edge_feats,
                                 lig_cfg.edge_hidden_size,
                                 lig_cfg.num_mpnn_layers)

        self.rec_gnns = get_gnns(rec_cfg.embed_size,
                                 rec_cfg.node_hidden_sizes,
                                 rec_edge_feats,
                                 rec_cfg.edge_hidden_size,
                                 rec_cfg.num_mpnn_layers)

        self.attentions = nn.ModuleList()
        self.lig_combiners = nn.ModuleList()
        for lig_sz, rec_sz in zip(lig_cfg.node_hidden_sizes, rec_cfg.node_hidden_sizes):
            # todo: this doesn't work when lig_size != rec_sz
            self.attentions.append(nn.MultiheadAttention(lig_sz, 1, kdim=rec_sz, vdim=rec_sz))
            self.lig_combiners.append(nn.Linear(lig_sz+rec_sz, lig_sz))

        assert len(self.lig_gnns) == len(self.rec_gnns)

        self.rec_readout = WeightedSumAndMax(rec_cfg.node_hidden_sizes[-1])
        self.lig_readout = WeightedSumAndMax(lig_cfg.node_hidden_sizes[-1])
        self.out_nns = nn.ModuleList()
        combined_hid_sz = 2*(rec_cfg.node_hidden_sizes[-1] + lig_cfg.node_hidden_sizes[-1])
            
        for prev, size in zip([combined_hid_sz] + list(cfg.model.out_mlp_sizes), cfg.model.out_mlp_sizes):
            self.out_nns.append(nn.Sequential(
                nn.Linear(prev, size),
                nn.LeakyReLU()
            ))
        self.out = nn.Linear(cfg.model.out_mlp_sizes[-1], 1)

    def forward(self, lig_graph, rec_graph):
        rec_hid = self.rec_embed(rec_graph.ndata['feat']).squeeze(1)
        lig_hid = lig_graph.ndata['feat']
        for lig_gnn, rec_gnn, atn, comb in zip(self.lig_gnns, self.rec_gnns, self.attentions, self.lig_combiners):
            rec_hid = rec_gnn(rec_graph, rec_hid, rec_graph.edata['feat'])
            lig_hid = lig_gnn(lig_graph, lig_hid, lig_graph.edata['feat'])

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
        return self.out(x)