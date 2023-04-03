from typing import List
from dataclassy import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.backend as dF
from dgl.nn import WeightAndSum
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

def is_tensor(obj):
    return isinstance(obj, torch.Tensor)

# def pad_packed_tensor(input, lengths, value, l_min=None):
#     old_shape = input.shape
#     device = input.device
#     if not is_tensor(lengths):
#         lengths = torch.tensor(lengths, dtype=torch.int64, device=device)
#     else:
#         lengths = lengths.to(device)
#     max_len = as_scalar(lengths.max())

#     if l_min is not None:
#         max_len = builtins.max(max_len, l_min)

#     batch_size = len(lengths)
#     x = input.new(batch_size * max_len, *old_shape[1:])
#     x.fill_(value)
#     index = torch.ones(len(input), dtype=torch.int64, device=device)
#     cum_lengths = torch.cumsum(lengths, 0)
#     index[cum_lengths[:-1]] += (max_len - lengths[:-1])
#     index = torch.cumsum(index, 0) - 1
#     x[index] = input
#     return x.view(batch_size, max_len, *old_shape[1:])

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
    input = input.view(-1, *input.shape[2:])
    return input[index]

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

    def forward(self, xy_feat, n_heads, n_feat, mask=None):
        self.start_forward()
        """ uses a sort of multi-head attention to contract an (X, Y, F) tensor and get
        a (X, n_heads*n_feat) tensor"""
        out_feat = self.make(NormAndLinear, self.cfg, n_heads*n_feat)(xy_feat).reshape((*xy_feat.shape[:-1], n_heads, n_feat))
        atn_coefs = self.make(LazyLinear, n_heads)(F.leaky_relu(xy_feat)).unsqueeze(-1)
        if mask is not None:
            atn_coefs = atn_coefs.masked_fill(mask.unsqueeze(-1), 0)

        atn_coefs = F.softmax(atn_coefs, -3)
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
        xy_feat = xy_feat.reshape((-1, xy_feat.shape[-1]))
        feat = self.make(NormAndLinear, self.cfg, n_heads*n_feat)(xy_feat).reshape((-1, n_heads, n_feat))
        atn = self.make(LazyLinear, n_heads)(F.leaky_relu(xy_feat)).unsqueeze(-1)
        if mask is not None:
            atn = atn.masked_fill(mask.reshape((-1, 1, 1)), 0)
        atn = atn.softmax(0)
        return (feat*atn).sum(0).reshape(-1)

# todo: both the following functions need to be optimized now that we
# are properly padding the ll feat tensors

def ll_feat_to_lig_edges(ll_feat, lig_graph):
    """ returns the data in ll_feat indexes by the edge indexes in lig_graph """
    all_src, all_dst = lig_graph.dgl().edges()
    all_src, all_dst = all_src.long(), all_dst.long()
    ret = []
    for llf, edge_slice, node_slice in zip(ll_feat, lig_graph.edge_slices, lig_graph.node_slices):
        src, dst = all_src[edge_slice] - node_slice.start, all_dst[edge_slice] - node_slice.start
        ret.append(llf[src, dst])
    return torch.cat(ret, 0)

def lig_edges_to_ll_feat(lig_edge_feat, lig_graph, L):
    all_src, all_dst = lig_graph.dgl().edges()
    all_src, all_dst = all_src.long(), all_dst.long()
    ret = []
    # L = lig_graph.dgl().batch_num_nodes().cpu().item()
    for edge_slice, node_slice in zip(lig_graph.edge_slices, lig_graph.node_slices):
        src, dst = all_src[edge_slice] - node_slice.start, all_dst[edge_slice] - node_slice.start
        llf = torch.zeros((L, L, lig_edge_feat.shape[-1]), device=lig_edge_feat.device)
        llf[src, dst] += lig_edge_feat[edge_slice]
        ret.append(llf)
    return torch.stack(ret)


def feat_to_edge_feat(x_feat, x_graph):
    all_src, all_dst = x_graph.dgl().edges()
    all_src, all_dst = all_src.long(), all_dst.long()
    return torch.cat((x_feat[all_src], x_feat[all_dst]), -1)

def cat_feat_list_list(feats):
    return [ torch.cat([feats[i][j] for i in range(len(feats))], -1) for j in range(len(feats[0])) ]

class TwistBlock(TwistModule):

    def full_attention_contract(self, xy_feat, n_heads, n_feat, graph_index, transpose=False, eye_mask=False):
        device = xy_feat.device
        if transpose:
            xy_feat = xy_feat.transpose(-3,-2)
        mask = torch.eye(xy_feat.shape[1], device=device, dtype=bool).unsqueeze(-1).unsqueeze(0) if eye_mask else None
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

        lig_packed = dF.pad_packed_tensor(td.lig_feat, twist_index.lig_lens, 0.0)
        rec_packed = dF.pad_packed_tensor(td.rec_feat, twist_index.rec_lens, 0.0)
        full_rec_packed = dF.pad_packed_tensor(td.full_rec_feat, twist_index.full_rec_lens, 0.0)

        # lig_feat, lig_edge_feat, ll_feat, l_ra_feat, l_rf_feat -> lig_feat
        lig_hid = [ td.lig_feat ] 
        lig_hid.append(self.make(MPNN)(x.lig_graph, td.lig_feat, td.lig_edge_feat))
        lig_hid.append(self.full_attention_contract(td.ll_feat, self.cfg.ll_atn_heads, self.cfg.ll_atn_feat, twist_index.lig_pad_index, eye_mask=True))
        lig_hid.append(self.full_attention_contract(td.l_ra_feat, self.cfg.l_ra_atn_heads, self.cfg.l_ra_atn_feat, twist_index.lig_pad_index))
        lig_hid.append(self.full_attention_contract(td.l_rf_feat, self.cfg.l_rf_atn_heads, self.cfg.l_rf_atn_feat, twist_index.lig_pad_index))

        lig_hid = torch.cat(lig_hid, -1)
        lig_feat = self.make(LazyLinear, td_cfg.lig_feat)(lig_hid)
        
        # rec_feat, rec_edge_feat, l_ra_feat, rf_feat -> rec_feat
        rec_hid = [ td.rec_feat ]
        rec_hid.append(self.make(MPNN)(x.rec_graph, td.rec_feat, td.rec_edge_feat))
        rec_hid.append(self.full_attention_contract(td.l_ra_feat, self.cfg.l_ra_atn_heads, self.cfg.l_ra_atn_feat, twist_index.rec_pad_index, transpose=True))
        rec_hid.append(self.make(FullRecContract, self.cfg)(td.full_rec_feat, twist_index.res_index, self.cfg.rf_ra_hid_size))

        rec_hid = torch.cat(rec_hid, -1)
        rec_feat = self.make(LazyLinear, td_cfg.rec_feat)(rec_hid)

        # full_rec_feat, l_rf_feat, ra_feat -> full_rec_feat
        full_rec_hid = [ td.full_rec_feat ]
        full_rec_hid.append(self.full_attention_contract(td.l_rf_feat, self.cfg.l_rf_atn_heads, self.cfg.l_rf_atn_feat, twist_index.full_rec_pad_index, transpose=True))
        full_rec_hid.append(self.make(FullRecExpand, self.cfg)(td.rec_feat, twist_index.res_index, self.cfg.rf_ra_hid_size))

        full_rec_hid = torch.cat(full_rec_hid, -1)
        full_rec_feat = self.make(LazyLinear, td_cfg.full_rec_feat)(full_rec_hid)

        # lig_edge_feat, lig_feat, ll_feat -> lig_edge_feat
        
        ll_2_lig_edge = self.make(NormAndLinear, self.cfg, self.cfg.ll_lig_edge_hid_size)(td.ll_feat)
        lig_2_lig_edge = self.make(NormAndLinear, self.cfg, self.cfg.lig_feat_lig_edge_hid_size)(td.lig_feat)

        lig_edge_hid = [ td.lig_edge_feat ]
        lig_edge_hid.append(feat_to_edge_feat(lig_2_lig_edge, x.lig_graph))
        lig_edge_hid.append(ll_feat_to_lig_edges(ll_2_lig_edge, x.lig_graph))
        
        lig_edge_hid = torch.cat(lig_edge_hid, -1)
        lig_edge_feat = self.make(LazyLinear, td_cfg.lig_edge_feat)(lig_edge_hid)

        # rec_edge_feat, rec_feat -> rec_edge_feat

        rec_2_rec_edge = self.make(NormAndLinear, self.cfg, self.cfg.rec_feat_rec_edge_hid_size)(td.rec_feat)

        rec_edge_hid = [ td.rec_edge_feat ]
        rec_edge_hid.append(feat_to_edge_feat(rec_2_rec_edge, x.rec_graph))        

        rec_edge_hid = torch.cat(rec_edge_hid, -1)
        rec_edge_feat = self.make(LazyLinear, td_cfg.rec_edge_feat)(rec_edge_hid)

        # ll_feat, lig_feat, lig_edge_feat -> ll_feat
        lig_edge_2_ll = self.make(NormAndLinear, self.cfg, self.cfg.ll_lig_edge_hid_size)(td.lig_edge_feat)

        ll_hid = [ td.ll_feat ]
        ll_hid.append(self.full_attention_expand(lig_packed, 
                                                lig_packed,
                                                self.cfg.ll_expand_heads,
                                                self.cfg.ll_expand_feat))
        ll_hid.append(lig_edges_to_ll_feat(lig_edge_2_ll, x.lig_graph, twist_index.Lm))

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
        for i in range(self.cfg.updates_per_block):
            td = self.update_once(x, twist_index, td)
        return td

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

    def flatten_xy_feat(self, all_xy_feat, n_heads, n_feat, eye_mask=False):
        device = all_xy_feat[0].device
        ret = []
        flatten = self.make(FlattenXYData, self.cfg)
        for xy_feat in all_xy_feat:
            mask = torch.eye(xy_feat.shape[0], device=device, dtype=bool).unsqueeze(-1) if eye_mask else None
            ret.append(flatten(xy_feat, n_heads, n_feat, mask))
        return torch.stack(ret)

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

    def forward(self, x):
        self.start_forward()

        # res_index = x.full_rec_data.get_res_index()
        # assert res_index.amax() + 1 == x.rec_graph.ndata.cat_feat.shape[0]
        twist_index = TwistIndex(x)

        td = self.make(TwistEncoder, self.cfg)(x, twist_index)
        for i in range(self.cfg.num_blocks):
            td = td + self.make(TwistBlock, self.cfg)(x, twist_index, td)

        scal_outputs = self.get_scal_outputs(td, x, 2)
        act = scal_outputs[:,0]
        is_act = scal_outputs[:,1]

        return Batch(DFRow,
                     score=is_act,
                     activity=act,
                     activity_score=act,
                     final_l_rf_hid=td.l_rf_feat,
                     final_l_ra_hid=td.l_ra_feat,
                     final_ll_hid=td.ll_feat)