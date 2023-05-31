from typing import List
from dataclassy import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import dgl.backend as dF
from dgl.nn import WeightAndSum
from dgl.nn.pytorch import EGATConv
from common.pose_transform import MultiPose, Pose
from models.attention_gnn import MPNN
from terrace import Module
from terrace.batch import Batch, Batchable
from terrace.dataframe import DFRow
from terrace.module import LazyLayerNorm, LazyLinear, LazyMultiheadAttention
from .model import ClassifyActivityModel
from .cat_scal_embedding import CatScalEmbedding
from .graph_embedding import GraphEmbedding
from .force_field import ScaleOutput, rbf_encode, cdist_diff

class EGATMPNN(Module):
    """ Very basic message passing step"""

    def __init__(self, out_node_feats, out_edge_feats, num_heads):
        super().__init__()
        self.out_node_feats = out_node_feats
        self.out_edge_feats = out_edge_feats
        self.num_heads = num_heads

    def forward(self, g, node_feats, edge_feats):
        self.start_forward()
        node_size = node_feats.shape[-1]
        edge_size = edge_feats.shape[-1]
        gnn = self.make(EGATConv, node_size, edge_size, self.out_node_feats, self.out_edge_feats, self.num_heads)
        ndata, edata = gnn(g.dgl(), F.leaky_relu(node_feats), F.leaky_relu(edge_feats))
        return ndata.reshape(ndata.shape[0], -1), edata.reshape(edata.shape[0], -1)

def is_tensor(obj):
    return isinstance(obj, torch.Tensor)

# taken from dgl backend code -- computing the index actually takes some time,
# so we precompute it

def get_padded_index(lengths):
    assert is_tensor(lengths)
    max_len = lengths.amax().item()
    out_len = lengths.sum().item()
    index = torch.ones(out_len, dtype=torch.int64, device=lengths.device)
    cum_lengths = torch.cumsum(lengths, 0)
    index[cum_lengths[:-1]] += (max_len - lengths[:-1])
    index = torch.cumsum(index, 0) - 1
    return index

def pack_padded_tensor_with_index(input, index):
    input = input.reshape(-1, *input.shape[2:])
    return input[index]

def pad_packed_tensor_with_index(input, lengths, max_len, index, value):
    old_shape = input.shape
    batch_size = len(lengths)
    x = input.new(batch_size * max_len, *old_shape[1:])
    x.fill_(value)
    x[index] = input
    return x.view(batch_size, max_len, *old_shape[1:])


def get_padded_edge_indexes(graph):
    """ Returns indexes mapping (padded) LL feat to (packed) lig edge feat"""
    all_src, all_dst = graph.dgl().edges()
    all_src, all_dst = all_src.long(), all_dst.long()
    batch = torch.zeros_like(all_src)
    src = torch.zeros_like(all_src)
    dst = torch.zeros_like(all_src)
    for i, (edge_slice, node_slice) in enumerate(zip(graph.edge_slices, graph.node_slices)):
        cur_src, cur_dst = all_src[edge_slice] - node_slice.start, all_dst[edge_slice] - node_slice.start
        batch[edge_slice] = i
        src[edge_slice] = cur_src
        dst[edge_slice] = cur_dst
    return batch, src, dst

def my_masked_fill(x, mask, val):
    """ super inefficient but works with jorch"""
    # return torch.masked_fill(x, mask, val)
    return x*(~mask) + ((torch.zeros_like(x)+val)*(mask))

def softmax_with_mask(x, mask, dim):
    # return torch.softmax(x, dim)
    x_exp = torch.exp(x - torch.max(x,dim=dim,keepdim=True)[0])
    x_exp = my_masked_fill(x_exp, ~mask, 0.0)
    ret = x_exp / (x_exp.sum(dim,keepdim=True) + 1e-10)
    return ret


class TwistIndex:
    """ Used to store precomputed indexes for padding/packing tensors """

    def __init__(self, x):
        self.lig_lens = x.lig_graph.dgl().batch_num_nodes()
        self.rec_lens = x.rec_graph.dgl().batch_num_nodes()
        self.full_rec_lens = x.full_rec_data.dgl().batch_num_nodes()

        self.lig_pad_index = get_padded_index(self.lig_lens)
        self.rec_pad_index = get_padded_index(self.rec_lens)
        self.full_rec_pad_index = get_padded_index(self.full_rec_lens)

        self.Lm = max(self.lig_lens).cpu().item()
        self.Ram = max(self.rec_lens).cpu().item()
        self.Rfm = max(self.full_rec_lens).cpu().item()

        self.res_index = x.full_rec_data.get_res_index()

        self.lig_edge_pad_index = get_padded_edge_indexes(x.lig_graph)

        lig_src, lig_dst = x.lig_graph.dgl().edges()
        self.lig_edge_index = lig_src.long(), lig_dst.long()

        rec_src, rec_dst = x.rec_graph.dgl().edges()
        self.rec_edge_index = rec_src.long(), rec_dst.long()


        device = x.lig_graph.ndata.cat_feat.device
        l_mask = dF.pad_packed_tensor(torch.ones((len(x.lig_graph.ndata,)), dtype=bool, device=device), x.lig_graph.dgl().batch_num_nodes(), 0.0)
        ra_mask = dF.pad_packed_tensor(torch.ones((len(x.rec_graph.ndata,)), dtype=bool, device=device), x.rec_graph.dgl().batch_num_nodes(), 0.0)
        rf_mask = dF.pad_packed_tensor(torch.ones((len(x.full_rec_data.ndata,)), dtype=bool, device=device), x.full_rec_data.dgl().batch_num_nodes(), 0.0)

        ll_mask = l_mask.unsqueeze(-1) & l_mask.unsqueeze(-2)
        ll_mask = ll_mask & ~torch.eye(ll_mask.shape[-1], dtype=bool, device=ll_mask.device).unsqueeze(0)

        l_ra_mask = l_mask.unsqueeze(-1) & ra_mask.unsqueeze(-2)
        l_rf_mask = l_mask.unsqueeze(-1) & rf_mask.unsqueeze(-2)

        self.ll_mask = ll_mask.unsqueeze(-1)
        self.l_ra_mask = l_ra_mask.unsqueeze(-1)
        self.l_rf_mask = l_rf_mask.unsqueeze(-1)

# L = number of atoms in ligand
# Ra = number of rec residues (nodes in Ca graph)
# Rf = number of rec atoms (ndoe in full graph)
# EL = number of edges in lig graph
# ER = number of edges in Ra graph
# *m = max number of nodes in graph *
# B = batch size
@dataclass
class TwistData:

    # these features are packed (no batch dim)
    lig_feat: torch.Tensor # (L, F)
    rec_feat: torch.Tensor # (Ra, F)
    full_rec_feat: torch.Tensor # (Rf, F)

    lig_edge_feat: torch.Tensor # (EL, F)
    rec_edge_feat: torch.Tensor # (ER, F)

    # these features are padded (batch dim, padding to max node number )
    ll_feat: torch.Tensor # (B, Lm, Lm, F)
    l_ra_feat: torch.Tensor # (B, Lm, Ram, F)
    l_rf_feat: torch.Tensor # (B, Lm, Rfm, F)


    def __add__(self, other):
        args = {}
        for key in self.__dict__.keys():
            val1 = getattr(self, key)
            val2 = getattr(other, key)
            if isinstance(val1, list):
                args[key] = [ v1 + v2 for v1, v2 in zip(val1, val2)]
            else:
                args[key] = val1 + val2
        return TwistData(**args)

class NormAndLinear(Module):

    def __init__(self, cfg, out_feat):
        super().__init__()
        self.out_feat = out_feat
        if cfg.normalization == "layer":
            self.norm_class = LazyLayerNorm
        else:
            raise ValueError(f"Unsupported normalization {self.cfg.normalization}")

    def forward(self, x):
        self.start_forward()
        hid = self.make(LazyLinear, self.out_feat)(F.leaky_relu(x))
        return (self.make(self.norm_class)(hid.reshape((-1, hid.shape[-1])))).reshape(hid.shape)

class TwistModule(Module):
    """ Base class for TwistEncoder and TwistBlock"""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

class FullNorm(TwistModule):

    def __init__(self, cfg):
        super().__init__(cfg)
        if cfg.normalization == "layer":
            self.norm_class = LazyLayerNorm
        else:
            raise ValueError(f"Unsupported normalization {self.cfg.normalization}")


    def forward(self, x):
        self.start_forward()
        args = {}
        for key in x.__dict__.keys():
            val = getattr(x, key)
            args[key] = (self.make(self.norm_class)(val.reshape((-1, val.shape[-1])))).reshape(val.shape)

        return TwistData(**args)

class TwistEncoder(TwistModule):
    """ Returns the initial TwistData object from the lig and rec graphs """

    def forward(self, x, twist_index):
        self.start_forward()

        td_cfg = self.cfg.twist_data

        device = x.lig_graph.ndata.cat_feat.device

        lig_feat, lig_edge_feat = self.make(GraphEmbedding, self.cfg.lig_encoder)(x.lig_graph)
        lig_feat = self.make(NormAndLinear, self.cfg, td_cfg.lig_feat)(lig_feat)
        lig_edge_feat = self.make(NormAndLinear, self.cfg, td_cfg.lig_edge_feat)(lig_edge_feat)

        rec_feat, rec_edge_feat = self.make(GraphEmbedding, self.cfg.rec_encoder)(x.rec_graph)
        rec_feat = self.make(NormAndLinear, self.cfg, td_cfg.rec_feat)(rec_feat)
        rec_edge_feat = self.make(NormAndLinear, self.cfg, td_cfg.rec_edge_feat)(rec_edge_feat)

        full_rec_feat = self.make(CatScalEmbedding, self.cfg.full_atom_embed_size)(x.full_rec_data.ndata)
        full_rec_feat = self.make(NormAndLinear, self.cfg, td_cfg.full_rec_feat)(full_rec_feat)

        B = len(x)

        ll_feat = torch.zeros((B, twist_index.Lm, twist_index.Lm, td_cfg.ll_feat), device=device)
        l_ra_feat = torch.zeros((B, twist_index.Lm, twist_index.Ram, td_cfg.l_ra_feat), device=device)
        l_rf_feat = torch.zeros((B, twist_index.Lm, twist_index.Rfm, td_cfg.l_rf_feat), device=device)


        return TwistData(
                    lig_feat=lig_feat,
                    rec_feat=rec_feat,
                    full_rec_feat=full_rec_feat,
                    lig_edge_feat=lig_edge_feat,
                    rec_edge_feat=rec_edge_feat,
                    ll_feat=ll_feat,
                    l_ra_feat=l_ra_feat,
                    l_rf_feat=l_rf_feat)

class AttentionContract(TwistModule):

    def forward(self, xy_feat, n_heads, n_feat, mask):
        self.start_forward()
        """ uses a sort of multi-head attention to contract an (X, Y, F) tensor and get
        a (X, n_heads*n_feat) tensor"""
        out_feat = self.make(NormAndLinear, self.cfg, n_heads*n_feat)(xy_feat).reshape((*xy_feat.shape[:-1], n_heads, n_feat))
        atn_coefs = self.make(LazyLinear, n_heads)(F.leaky_relu(xy_feat)).unsqueeze(-1)
        atn_coefs = softmax_with_mask(atn_coefs, mask.unsqueeze(-1), -3)
        out = (atn_coefs*out_feat).sum(-3).reshape((*atn_coefs.shape[:-3], -1))
        return out

class AttentionExpand(TwistModule):

    def forward(self, x_feat, y_feat, n_heads, n_feat):
        self.start_forward()
        """ uses a sort of mult-head attention to expand a (X, F1) tensor and a (Y, F2) tensor
        to a (X, Y, n_heads) tensor"""
        x_feat = self.make(NormAndLinear, self.cfg, n_heads*n_feat)(x_feat).reshape((*x_feat.shape[:-1], n_heads, n_feat))
        y_feat = self.make(NormAndLinear, self.cfg, n_heads*n_feat)(y_feat).reshape((*y_feat.shape[:-1], n_heads, n_feat))
        out = torch.einsum("...xhf,...yhf->...xyh", x_feat, y_feat)
        return out

class FullRecContract(TwistModule):

    def forward(self, full_rec_data, res_index, n_feat):
        """ Contracts (Rf, F1) to (Ra, n_feat) tensor"""
        self.start_forward()
        feat = self.make(NormAndLinear, self.cfg, n_feat)(full_rec_data)
        
        # use dgl backend scatter_sum to softmax over residue indexes
        # alas this isn't documented in dgl but torch_scatter is so
        # annoying to install
        atn = self.make(LazyLinear, 1)(F.leaky_relu(full_rec_data))
        atn = atn - atn.amax()
        atn = torch.exp(atn)
        atn_sum = dF.scatter_add(atn, res_index, res_index.amax()+1)
        atn_sum_expand = atn_sum[res_index]
        atn = atn/atn_sum_expand

        return dF.scatter_add(feat*atn, res_index, res_index.amax()+1)

class FullRecExpand(TwistModule):

    def forward(self, rec_data, res_index, n_feat):
        """ Expands (Ra, F1) to (Rf, n_feat) """
        self.start_forward()
        feat = self.make(NormAndLinear, self.cfg, n_feat)(rec_data)
        return feat[res_index]

class FlattenXYData(TwistModule):
    """ Flattens all the 2D data (ll_feat, l_ra_feat, etc) into a single tensor
    using weighted sums with a certain number of heads """

    def forward(self, xy_feat, n_heads, n_feat, mask=None):
        self.start_forward()
        B = xy_feat.shape[0]
        xy_feat = xy_feat.reshape((B, -1, xy_feat.shape[-1]))
        feat = self.make(NormAndLinear, self.cfg, n_heads*n_feat)(xy_feat).reshape((B, -1, n_heads, n_feat))
        atn = self.make(LazyLinear, n_heads)(F.leaky_relu(xy_feat)).unsqueeze(-1)
        atn = softmax_with_mask(atn, mask.reshape((1, -1, 1, 1)), 1)
        raise Exception("debug me!")
        return (feat*atn).sum(1).reshape(B, -1)

def ll_feat_to_lig_edges(ll_feat, lig_edge_index):
    """ returns the data in ll_feat indexes by the edge indexes in lig_graph """
    return ll_feat[lig_edge_index]

def lig_edges_to_ll_feat(lig_edge_feat, lig_edge_index, B, L):
    llf = torch.zeros((B, L, L, lig_edge_feat.shape[-1]), device=lig_edge_feat.device)
    llf[lig_edge_index] += lig_edge_feat
    return llf

def feat_to_edge_feat(x_feat, edge_index):
    src, dst = edge_index
    return torch.cat((x_feat[src], x_feat[dst]), -1)

def cat_feat_list_list(feats):
    return [ torch.cat([feats[i][j] for i in range(len(feats))], -1) for j in range(len(feats[0])) ]

class TwistBlock(TwistModule):

    def full_attention_contract(self, xy_feat, n_heads, n_feat, graph_index, mask, transpose=False):
        if transpose:
            xy_feat = xy_feat.transpose(-3,-2)
            mask = mask.transpose(-3,-2)
        ret = self.make(AttentionContract, self.cfg)(xy_feat, n_heads, n_feat, mask)
        return pack_padded_tensor_with_index(ret, graph_index)

    def full_attention_expand(self, x_feat, y_feat, n_heads, n_feat):
        return self.make(AttentionExpand, self.cfg)(x_feat, y_feat, n_heads, n_feat)

    def full_norm_and_linear(self, feat_list, out_feat):
        """ Runs a new NormAndLinear layer on an list of tensors"""
        mod = self.make(NormAndLinear, self.cfg, out_feat)
        return [ mod(feat) for feat in feat_list ]

    def full_linear(self, feat_list, out_feat):
        """ Runs a new NormAndLinear layer on an list of tensors"""
        mod = self.make(LazyLinear, out_feat)
        return [ mod(feat) for feat in feat_list ]

    def update_once(self, x, twist_index, td):

        td_cfg = self.cfg.twist_data

        lig_packed = pad_packed_tensor_with_index(td.lig_feat, twist_index.lig_lens, twist_index.Lm, twist_index.lig_pad_index, 0.0)
        rec_packed = pad_packed_tensor_with_index(td.rec_feat, twist_index.rec_lens, twist_index.Ram, twist_index.rec_pad_index,  0.0)
        full_rec_packed = pad_packed_tensor_with_index(td.full_rec_feat, twist_index.full_rec_lens, twist_index.Rfm, twist_index.full_rec_pad_index,  0.0)

        mpnn_type = self.cfg.get("mpnn_type", "default")
        if mpnn_type == "default":
            lig_mpnn_hid = self.make(MPNN)(x.lig_graph, td.lig_feat, td.lig_edge_feat)
            rec_mpnn_hid = self.make(MPNN)(x.rec_graph, td.rec_feat, td.rec_edge_feat)
            lig_edge_mpnn_hid = None
            rec_edge_mpnn_hid = None
        elif mpnn_type == "egat":
            lig_mpnn_hid, lig_edge_mpnn_hid = self.make(EGATMPNN, self.cfg.lig_mpnn_node_size, self.cfg.lig_mpnn_edge_size, self.cfg.lig_mpnn_heads)(x.lig_graph, td.lig_feat, td.lig_edge_feat)
            rec_mpnn_hid, rec_edge_mpnn_hid = self.make(EGATMPNN, self.cfg.rec_mpnn_node_size, self.cfg.rec_mpnn_edge_size, self.cfg.rec_mpnn_heads)(x.rec_graph, td.rec_feat, td.rec_edge_feat)

        # lig_feat, lig_edge_feat, ll_feat, l_ra_feat, l_rf_feat -> lig_feat
        lig_hid = [ td.lig_feat ] 
        lig_hid.append(lig_mpnn_hid)
        lig_hid.append(self.full_attention_contract(td.ll_feat, self.cfg.ll_atn_heads, self.cfg.ll_atn_feat, twist_index.lig_pad_index, twist_index.ll_mask))
        lig_hid.append(self.full_attention_contract(td.l_ra_feat, self.cfg.l_ra_atn_heads, self.cfg.l_ra_atn_feat, twist_index.lig_pad_index, twist_index.l_ra_mask))
        lig_hid.append(self.full_attention_contract(td.l_rf_feat, self.cfg.l_rf_atn_heads, self.cfg.l_rf_atn_feat, twist_index.lig_pad_index, twist_index.l_rf_mask))

        lig_hid = torch.cat(lig_hid, -1)
        lig_feat = self.make(LazyLinear, td_cfg.lig_feat)(lig_hid)
        
        # rec_feat, rec_edge_feat, l_ra_feat, rf_feat -> rec_feat
        rec_hid = [ td.rec_feat ]
        rec_hid.append(rec_mpnn_hid)
        rec_hid.append(self.full_attention_contract(td.l_ra_feat, self.cfg.l_ra_atn_heads, self.cfg.l_ra_atn_feat, twist_index.rec_pad_index, twist_index.l_ra_mask, transpose=True))
        rec_hid.append(self.make(FullRecContract, self.cfg)(td.full_rec_feat, twist_index.res_index, self.cfg.rf_ra_hid_size))

        rec_hid = torch.cat(rec_hid, -1)
        rec_feat = self.make(LazyLinear, td_cfg.rec_feat)(rec_hid)

        # full_rec_feat, l_rf_feat, ra_feat -> full_rec_feat
        full_rec_hid = [ td.full_rec_feat ]
        full_rec_hid.append(self.full_attention_contract(td.l_rf_feat, self.cfg.l_rf_atn_heads, self.cfg.l_rf_atn_feat, twist_index.full_rec_pad_index, twist_index.l_rf_mask, transpose=True))
        full_rec_hid.append(self.make(FullRecExpand, self.cfg)(td.rec_feat, twist_index.res_index, self.cfg.rf_ra_hid_size))

        full_rec_hid = torch.cat(full_rec_hid, -1)
        full_rec_feat = self.make(LazyLinear, td_cfg.full_rec_feat)(full_rec_hid)

        # lig_edge_feat, lig_feat, ll_feat -> lig_edge_feat
        
        ll_2_lig_edge = self.make(NormAndLinear, self.cfg, self.cfg.ll_lig_edge_hid_size)(td.ll_feat)
        lig_2_lig_edge = self.make(NormAndLinear, self.cfg, self.cfg.lig_feat_lig_edge_hid_size)(td.lig_feat)

        lig_edge_hid = [ td.lig_edge_feat ]
        if lig_edge_mpnn_hid is not None:
            lig_edge_hid.append(lig_edge_mpnn_hid)
        lig_edge_hid.append(feat_to_edge_feat(lig_2_lig_edge, twist_index.lig_edge_index))
        lig_edge_hid.append(ll_feat_to_lig_edges(ll_2_lig_edge, twist_index.lig_edge_pad_index))
        
        lig_edge_hid = torch.cat(lig_edge_hid, -1)
        lig_edge_feat = self.make(LazyLinear, td_cfg.lig_edge_feat)(lig_edge_hid)

        # rec_edge_feat, rec_feat -> rec_edge_feat

        rec_2_rec_edge = self.make(NormAndLinear, self.cfg, self.cfg.rec_feat_rec_edge_hid_size)(td.rec_feat)

        rec_edge_hid = [ td.rec_edge_feat ]
        if rec_edge_mpnn_hid is not None:
            rec_edge_hid.append(rec_edge_mpnn_hid)
        rec_edge_hid.append(feat_to_edge_feat(rec_2_rec_edge, twist_index.rec_edge_index))        

        rec_edge_hid = torch.cat(rec_edge_hid, -1)
        rec_edge_feat = self.make(LazyLinear, td_cfg.rec_edge_feat)(rec_edge_hid)

        # ll_feat, lig_feat, lig_edge_feat -> ll_feat
        lig_edge_2_ll = self.make(NormAndLinear, self.cfg, self.cfg.ll_lig_edge_hid_size)(td.lig_edge_feat)

        ll_hid = [ td.ll_feat ]
        ll_hid.append(self.full_attention_expand(lig_packed, 
                                                lig_packed,
                                                self.cfg.ll_expand_heads,
                                                self.cfg.ll_expand_feat))
        ll_hid.append(lig_edges_to_ll_feat(lig_edge_2_ll, twist_index.lig_edge_pad_index, td.ll_feat.shape[0], td.ll_feat.shape[1]))

        # ll_hid = cat_feat_list_list(ll_hid)
        # ll_feat = self.full_linear(ll_hid, td_cfg.ll_feat)
        ll_hid = torch.cat(ll_hid, -1)
        ll_feat = self.make(LazyLinear, td_cfg.ll_feat)(ll_hid)

        # l_ra_feat, lig_feat, rec_feat -> l_ra_feat
        
        l_ra_hid = [ td.l_ra_feat ]
        l_ra_hid.append(self.full_attention_expand(lig_packed, 
                                                rec_packed,
                                                self.cfg.l_ra_expand_heads,
                                                self.cfg.l_ra_expand_feat))
        # l_ra_hid = cat_feat_list_list(l_ra_hid)
        # l_ra_feat = self.full_linear(l_ra_hid, td_cfg.l_ra_feat)
        
        l_ra_hid = torch.cat(l_ra_hid, -1)
        l_ra_feat = self.make(LazyLinear, td_cfg.l_ra_feat)(l_ra_hid)

        # l_rf_feat, lig_feat, full_rec_feat -> l_rf_feat
        
        l_rf_hid = [ td.l_rf_feat ]
        l_rf_hid.append(self.full_attention_expand(lig_packed, 
                                                full_rec_packed,
                                                self.cfg.l_rf_expand_heads,
                                                self.cfg.l_rf_expand_feat))
        # l_rf_hid = cat_feat_list_list(l_rf_hid)
        # l_rf_feat = self.full_linear(l_rf_hid, td_cfg.l_rf_feat)
        
        l_rf_hid = torch.cat(l_rf_hid, -1)
        l_rf_feat = self.make(LazyLinear, td_cfg.l_rf_feat)(l_rf_hid)

        return TwistData(
            lig_feat=lig_feat,
            rec_feat=rec_feat,
            full_rec_feat=full_rec_feat,
            lig_edge_feat=lig_edge_feat,
            rec_edge_feat=rec_edge_feat,
            ll_feat=ll_feat,
            l_ra_feat=l_ra_feat,
            l_rf_feat=l_rf_feat
        )

    def forward(self, x, twist_index, td):
        self.start_forward()
        for i in range(self.cfg.updates_per_block - 1):
            td = self.update_once(x, twist_index, td)
            if self.cfg.get("norm_after_add", False):
                td = self.make(FullNorm, self.cfg)(td)
        return self.update_once(x, twist_index, td)


class TwisterV2(Module, ClassifyActivityModel):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.model

    @staticmethod
    def get_name():
        return "twister_v2"

    def get_tasks(self):
        return [ "score_activity_class",
                 "classify_activity",
                 "predict_activity",
                 "score_activity_regr" 
                ]

    def get_input_feats(self):
        return [ "lig_graph", "rec_graph", "full_rec_data" ]

    def flatten_xy_feat(self, xy_feat, n_heads, n_feat, eye_mask=False):
        device = xy_feat[0].device
        mask = torch.eye(xy_feat.shape[1], device=device, dtype=bool).unsqueeze(-1) if eye_mask else None
        return self.make(FlattenXYData, self.cfg)(xy_feat, n_heads, n_feat, mask)

    def get_scal_outputs(self, td, x, num_outputs):
        """ Return the scalar outputs (activity, is_active, etc) """
        lig_hid = self.make(NormAndLinear, self.cfg, self.cfg.lig_scal_hid)(td.lig_feat)
        lig_flat = self.make(WeightAndSum, lig_hid.shape[-1])(x.lig_graph.dgl(), lig_hid)

        rec_hid = self.make(NormAndLinear, self.cfg, self.cfg.rec_scal_hid)(td.rec_feat)
        rec_flat = self.make(WeightAndSum, rec_hid.shape[-1])(x.rec_graph.dgl(), rec_hid)

        full_rec_hid = self.make(NormAndLinear, self.cfg, self.cfg.full_rec_scal_hid)(td.full_rec_feat)
        full_rec_flat = self.make(WeightAndSum, full_rec_hid.shape[-1])(x.full_rec_data.dgl(), full_rec_hid)

        # add in ll, l_ra, and l_rf feats
        ll_flat = self.flatten_xy_feat(td.ll_feat, self.cfg.ll_scal_heads, self.cfg.ll_scal_hid, True)
        l_ra_flat = self.flatten_xy_feat(td.l_ra_feat, self.cfg.l_ra_scal_heads, self.cfg.l_ra_scal_hid)
        l_rf_flat = self.flatten_xy_feat(td.l_rf_feat, self.cfg.l_rf_scal_heads, self.cfg.l_rf_scal_hid)

        hid = torch.cat([ lig_flat, rec_flat, full_rec_flat, ll_flat, l_ra_flat, l_rf_flat ], -1)
        hid = self.make(NormAndLinear, self.cfg, self.cfg.scal_hid)(hid)
        scal = self.make(LazyLinear, num_outputs)(F.leaky_relu(hid))
        return scal

    def get_inv_dist_mat(self, td):
        return self.make(LazyLinear, 1)(F.leaky_relu(td.l_rf_feat))[...,0]

    def forward(self, x):
        self.start_forward()

        twist_index = TwistIndex(x)

        td = self.make(TwistEncoder, self.cfg)(x, twist_index)
        for i in range(self.cfg.num_blocks):
            td = td + self.make(TwistBlock, self.cfg)(x, twist_index, td)
            if self.cfg.get("norm_after_add", False):
                td = self.make(FullNorm, self.cfg)(td)

        scal_outputs = self.get_scal_outputs(td, x, 2)
        act = scal_outputs[:,0]
        is_act = scal_outputs[:,1]

        # inv_dist_mat = self.get_inv_dist_mat(td)

        return Batch(DFRow,
                     score=is_act,
                     activity=act,
                     activity_score=act,
                     # inv_dist_mat=inv_dist_mat,
                    )
                    #  final_l_rf_hid=td.l_rf_feat,
                    #  final_l_ra_hid=td.l_ra_feat,
                    #  final_ll_hid=td.ll_feat)

# need to redefine some basic pytorch functions so we can automagically jaxify them
def run_leaky_relu(x, neg_slope=0.01):
    mask = x > 0
    return x*mask + neg_slope*x*(~mask)

def run_linear(x, wb):
    weight, bias = wb
    og_shape = x.shape[:-1]
    out = bias + x.reshape((-1, x.shape[-1])).matmul(weight.T)
    return out.reshape((*og_shape, -1))

def run_attention_collapse(x, heads, y_wb, atn_wb, mask):
    y = run_linear(x, y_wb).reshape((*x.shape[:-1], -1, heads))
    atn = run_linear(x, atn_wb).reshape((*x.shape[:-1], 1, heads))
    atn = softmax_with_mask(atn, mask.unsqueeze(-1).unsqueeze(-1), -3)
    ret = (y*atn).sum(-3)
    return run_leaky_relu(ret.reshape((*ret.shape[:-2], -1)))

class TwistFFCoef(Batchable):
    l_rf_coef: torch.Tensor
    ll_coef: torch.Tensor
    inv_dist_mat: torch.Tensor

class TwistForceField(TwistModule):

    def get_input_feats(self):
        return [ "lig_graph", "rec_graph", "full_rec_data" ]

    def __init__(self, cfg):
        super().__init__(cfg)
        if "energy" in self.cfg:
            self.l_rf_out = nn.Linear(self.cfg.rbf_steps, self.cfg.energy.l_rf_heads*self.cfg.energy.l_rf_feat)
            self.l_rf_atn = nn.Linear(self.cfg.rbf_steps, self.cfg.energy.l_rf_heads)

            self.ll_out = nn.Linear(self.cfg.rbf_steps, self.cfg.energy.ll_heads*self.cfg.energy.ll_feat)
            self.ll_atn = nn.Linear(self.cfg.rbf_steps, self.cfg.energy.ll_heads)

            hid_size = self.cfg.energy.ll_heads*self.cfg.energy.ll_feat + self.cfg.energy.l_rf_heads*self.cfg.energy.l_rf_feat
            self.out_lin = nn.Linear(hid_size, 1)

            single_hid_size = self.cfg.energy.single_heads*self.cfg.energy.single_feat
            self.single_head = nn.Linear(hid_size, single_hid_size)
            self.single_atn = nn.Linear(hid_size, self.cfg.energy.single_heads)
            self.single_hid = nn.Linear(single_hid_size, self.cfg.energy.single_hid)
            self.single_out = nn.Linear(self.cfg.energy.single_hid, 2)


    def get_hidden_feat(self, x):
        # smh autocast causes NaNs!
        # with autocast():
        self.start_forward()

        twist_index = TwistIndex(x)

        td = self.make(TwistEncoder, self.cfg)(x, twist_index)
        for i in range(self.cfg.num_blocks):
            td = td + self.make(TwistBlock, self.cfg)(x, twist_index, td)
            if self.cfg.get("norm_after_add", False):
                td = self.make(FullNorm, self.cfg)(td)

        self.scale_output = self.make(ScaleOutput, self.cfg.energy_bias)
        self.inv_dist_out = self.make(LazyLinear, 1)

        return Batch(TwistFFCoef,
            l_rf_coef = self.make(LazyLinear, self.cfg.rbf_steps)(F.leaky_relu(td.l_rf_feat)),
            ll_coef = self.make(LazyLinear, self.cfg.rbf_steps)(F.leaky_relu(td.ll_feat)),
            inv_dist_mat = self.make(LazyLinear, 1)(F.leaky_relu(td.l_rf_feat))[...,0]
        )

    @staticmethod
    def static_get_energy(mcfg,
                        coef,
                        lig_pose,
                        lig_graph,
                        full_rec_data,
                        weight,
                        bias):

        l_rf_coef = coef.l_rf_coef
        ll_coef = coef.ll_coef

        lig_coord = torch.cat(lig_pose.coord, -2)
        if lig_coord.dim() == 3:
            lig_coord = lig_coord.transpose(0,1)
        lig_coord = dF.pad_packed_tensor(lig_coord, lig_graph.dgl().batch_num_nodes(), 0.0)
        rec_coord = dF.pad_packed_tensor(full_rec_data.ndata.coord, full_rec_data.dgl().batch_num_nodes(), 0.0)
        if lig_coord.dim() == 4:
            lig_coord = lig_coord.transpose(1,2)
            rec_coord = rec_coord.unsqueeze(1)
            l_rf_coef = l_rf_coef.unsqueeze(1)
            ll_coef = ll_coef.unsqueeze(1)

        l_rf_dist = cdist_diff(lig_coord, rec_coord)
        ll_dist = cdist_diff(lig_coord, lig_coord)

        l_rf_mask = (lig_coord[...,0] != 0.0).unsqueeze(-1) & (rec_coord[...,0] != 0.0).unsqueeze(-2)
        ll_mask =  (lig_coord[...,0] != 0.0).unsqueeze(-1) & (lig_coord[...,0] != 0.0).unsqueeze(-2)

        extra_ll = ~torch.eye(ll_mask.shape[-1], dtype=bool, device=ll_mask.device).unsqueeze(0)
        if lig_coord.dim() == 4:
            extra_ll = extra_ll.unsqueeze(1)
        ll_mask = ll_mask & extra_ll

        l_rf_rbfs = rbf_encode(l_rf_dist, mcfg.rbf_start,mcfg.rbf_end,mcfg.rbf_steps)
        ll_rbfs = rbf_encode(ll_dist, mcfg.rbf_start,mcfg.rbf_end,mcfg.rbf_steps)

        l_rf_U = (l_rf_rbfs*l_rf_coef)
        l_rf_U[~l_rf_mask] = 0.0
        ll_U = (ll_rbfs*ll_coef)
        ll_U[~ll_mask] = 0.0

        pred_dist = (l_rf_U.sum((-1,-2)) + ll_U.sum((-1,-2)))*weight + bias
        if pred_dist.dim() == 3:
            pred_dist = pred_dist.transpose(-1,-2)
        idx = get_padded_index(lig_graph.dgl().batch_num_nodes())
        pred_dist = pack_padded_tensor_with_index(pred_dist,idx)
        if pred_dist.dim() == 2:
            pred_dist = pred_dist.transpose(-1,-2)

        pred_dist = dF.pad_packed_tensor(pred_dist if pred_dist.dim() == 1 else pred_dist.transpose(0,1), lig_graph.dgl().batch_num_nodes(), 0.0)
        U = (l_rf_U.sum((-1,-2,-3)) + ll_U.sum((-1,-2,-3)))*weight + bias
        return Batch(DFRow, dist=pred_dist, energy=U)

    @staticmethod
    def static_get_energy_v2(mcfg,
                            coef,
                            lig_pose,
                            lig_graph,
                            full_rec_data,
                            *params):

        params = TwistForceField.unflatten_params(params)

        l_rf_coef = coef.l_rf_coef
        ll_coef = coef.ll_coef

        lig_coord = torch.cat(lig_pose.coord, -2)
        if lig_coord.dim() == 3:
            lig_coord = lig_coord.transpose(0,1)
        lig_coord = dF.pad_packed_tensor(lig_coord, lig_graph.dgl().batch_num_nodes(), 0.0)
        rec_coord = dF.pad_packed_tensor(full_rec_data.ndata.coord, full_rec_data.dgl().batch_num_nodes(), 0.0)
        if lig_coord.dim() == 4:
            lig_coord = lig_coord.transpose(1,2)
            rec_coord = rec_coord.unsqueeze(1)
            l_rf_coef = l_rf_coef.unsqueeze(1)
            ll_coef = ll_coef.unsqueeze(1)

        l_rf_dist = cdist_diff(lig_coord, rec_coord)
        ll_dist = cdist_diff(lig_coord, lig_coord)

        l_rf_mask = (lig_coord[...,0] != 0.0).unsqueeze(-1) & (rec_coord[...,0] != 0.0).unsqueeze(-2)
        ll_mask =  (lig_coord[...,0] != 0.0).unsqueeze(-1) & (lig_coord[...,0] != 0.0).unsqueeze(-2)

        extra_ll = ~torch.eye(ll_mask.shape[-1], dtype=bool, device=ll_mask.device).unsqueeze(0)
        if lig_coord.dim() == 4:
            extra_ll = extra_ll.unsqueeze(1)
        ll_mask = ll_mask & extra_ll

        l_rf_rbfs = rbf_encode(l_rf_dist, mcfg.rbf_start,mcfg.rbf_end,mcfg.rbf_steps)
        ll_rbfs = rbf_encode(ll_dist, mcfg.rbf_start,mcfg.rbf_end,mcfg.rbf_steps)

        l_rf_data = l_rf_rbfs*l_rf_coef
        ll_data = ll_rbfs*ll_coef

        l_rf_col = run_attention_collapse(l_rf_data, mcfg.energy.l_rf_heads, params.l_rf_y_wb, params.l_rf_atn_wb, l_rf_mask)
        ll_col = run_attention_collapse(ll_data, mcfg.energy.ll_heads, params.ll_y_wb, params.ll_atn_wb, ll_mask)
        comb = torch.cat((ll_col, l_rf_col), -1)

        out = run_linear(comb, params.out_wb).squeeze(-1)
        pred_dist = out*params.final_weight + params.final_bias
        pred_dist = pred_dist.transpose(1,2).contiguous()

        single_hid = run_attention_collapse(comb, mcfg.energy.single_heads, params.single_head, params.single_atn, ll_mask[...,0])
        single_hid = run_linear(single_hid, params.single_hid)
        single_hid = run_leaky_relu(single_hid)
        single_out = run_linear(single_hid, params.single_out)

        pred_noise = single_out[...,0]
        pred_rmsd = single_out[...,0]

        energy = pred_dist.mean(-2)

        return Batch(DFRow, dist=pred_dist, rmsd=pred_rmsd, noise=pred_noise, energy=energy)

    @staticmethod
    def get_energy_single_v1(mcfg,
                        ll_coef,
                        l_rf_coef,
                        rec_coord,
                        lig_coord,
                        weight,
                        bias):
        
        l_rf_dist = cdist_diff(lig_coord, rec_coord)
        ll_dist = cdist_diff(lig_coord, lig_coord)
        ll_mask = ~torch.eye(ll_dist.shape[0], dtype=bool, device=ll_dist.device).unsqueeze(-1)
        

        l_rf_rbfs = rbf_encode(l_rf_dist, mcfg.rbf_start,mcfg.rbf_end,mcfg.rbf_steps)
        ll_rbfs = rbf_encode(ll_dist, mcfg.rbf_start,mcfg.rbf_end,mcfg.rbf_steps)

        ll_coef = ll_coef[:ll_rbfs.shape[0],:ll_rbfs.shape[1]]
        l_rf_coef = l_rf_coef[:l_rf_rbfs.shape[0],:l_rf_rbfs.shape[1]]

        l_rf_U = (l_rf_rbfs*l_rf_coef)
        ll_U = (ll_rbfs*ll_coef)
        # ll_U = torch.where(ll_mask, ll_U, torch.tensor(0.0, device=l_rf_dist.device))

        dist_U = (l_rf_U.sum((-1,-2)) + ll_U.sum((-1,-2)))*weight + bias
        # dist_U = dist_U.where(dist_U < -100.0, torch.tensor(0.0, device=l_rf_dist.device))

        # print(ll_U[0][0])
        # print(dist_U.where(dist_U < -100.0, torch.tensor(0.0, device=l_rf_dist.device)))

        # return torch.sqrt((dist_U**2).sum())
        return dist_U.mean()

    @staticmethod
    def get_energy_single_v2(mcfg,
                        ll_coef,
                        l_rf_coef,
                        rec_coord,
                        lig_coord,
                        *params):
        
        params = TwistForceField.unflatten_params(params)

        l_rf_dist = cdist_diff(lig_coord, rec_coord)
        ll_dist = cdist_diff(lig_coord, lig_coord)

        ll_mask = ~torch.eye(ll_dist.shape[0], dtype=bool, device=ll_dist.device)# .unsqueeze(-1)
        l_rf_mask = torch.zeros((*l_rf_dist.shape,), dtype=bool, device=ll_dist.device)

        l_rf_rbfs = rbf_encode(l_rf_dist, mcfg.rbf_start,mcfg.rbf_end,mcfg.rbf_steps)
        ll_rbfs = rbf_encode(ll_dist, mcfg.rbf_start,mcfg.rbf_end,mcfg.rbf_steps)

        l_rf_data = l_rf_rbfs*l_rf_coef
        ll_data = ll_rbfs*ll_coef

        l_rf_col = run_attention_collapse(l_rf_data, mcfg.energy.l_rf_heads, params.l_rf_y_wb, params.l_rf_atn_wb, l_rf_mask)
        ll_col = run_attention_collapse(ll_data, mcfg.energy.ll_heads, params.ll_y_wb, params.ll_atn_wb, ll_mask)
        comb = torch.cat((ll_col, l_rf_col), -1)

        out = run_linear(comb, params.out_wb).squeeze(-1)
        return (out*params.final_weight + params.final_bias).sum()

    @staticmethod
    def get_energy_single(mcfg, *args):
        if "energy" in mcfg:
            return TwistForceField.get_energy_single_v2(mcfg, *args)
        else:
            return TwistForceField.get_energy_single_v1(mcfg, *args)

    def get_energy(self, x, coef, lig_pose):
        if "energy" in self.cfg:
            return self.get_energy_v2(x, coef, lig_pose)
        bias, weight = self.scale_output.parameters()
        return TwistForceField.static_get_energy(self.cfg, coef, lig_pose, x.lig_graph, x.full_rec_data, weight, bias)

    def get_energy_v2_params(self):
        return [
           *tuple(self.l_rf_out.parameters()),
           *tuple(self.l_rf_atn.parameters()),
           *tuple(self.ll_out.parameters()),
           *tuple(self.ll_atn.parameters()),
           *tuple(self.out_lin.parameters()),
           *tuple(self.scale_output.parameters()),
           *tuple(self.single_head.parameters()),
           *tuple(self.single_atn.parameters()),
           *tuple(self.single_hid.parameters()),
           *tuple(self.single_out.parameters()),
        ]

    @staticmethod
    def unflatten_params(params):
        it = iter(params)
        return DFRow(
            l_rf_y_wb=(next(it), next(it)),
            l_rf_atn_wb=(next(it), next(it)),
            ll_y_wb=(next(it), next(it)),
            ll_atn_wb=(next(it), next(it)),
            out_wb=(next(it), next(it)),
            final_bias=next(it),
            final_weight=next(it),
            single_head=(next(it), next(it)),
            single_atn=(next(it), next(it)),
            single_hid=(next(it), next(it)),
            single_out=(next(it), next(it)),
        )

    def get_energy_v2(self, x, coef, lig_pose):
        params = self.get_energy_v2_params()
        return TwistForceField.static_get_energy_v2(self.cfg, coef, lig_pose, x.lig_graph, x.full_rec_data, *params)