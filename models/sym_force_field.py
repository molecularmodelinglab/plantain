import torch
import torch.nn as nn
import torch.nn.functional as F
from common.pose_transform import Pose
from models.attention_gnn import MPNN
from terrace import Module
from terrace.batch import Batch
from terrace.dataframe import DFRow
from terrace.module import LazyLayerNorm, LazyLinear, LazyMultiheadAttention
from .model import ClassifyActivityModel
from .cat_scal_embedding import CatScalEmbedding
from .graph_embedding import GraphEmbedding
from .force_field import ScaleOutput, rbf_encode, cdist_diff

class SymForceField(Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.model

    def get_input_feats(self):
        ret = [ "lig_graph", "rec_graph" ]
        if self.cfg.project_full_atom:
            ret.append("full_rec_data")
        return ret

    def make_normalization(self):
        if self.cfg.normalization == "layer":
            return self.make(LazyLayerNorm)
        raise ValueError(f"Unsupported normalization {self.cfg.normalization}")

    def run_linear(self, out_size, hid):
        hid = self.make(LazyLinear, out_size)(F.leaky_relu(hid))
        return self.make_normalization()(hid)

    def get_hidden_feat(self, enc_cfg, graph):
        assert not self.cfg.inner_attention
        node_feats, edge_feats = self.make(GraphEmbedding, enc_cfg)(graph)
        hid = self.run_linear(enc_cfg.node_out_size, node_feats)

        prev_hid = []
        for layer in range(self.cfg.num_mpnn_layers):
            hid = self.make(MPNN)(graph, F.leaky_relu(hid), edge_feats)
            hid = self.make_normalization()(hid)
            prev_layer = layer - 2
            if prev_layer >= 0:
                hid = hid + prev_hid[prev_layer]
            prev_hid.append(hid)

        return hid

    def get_interaction_matrix(self, x):
        self.start_forward()

        lig_hid = None
        if hasattr(x, "lig_graph"):
            lig_hid = self.get_hidden_feat(self.cfg.lig_encoder, x.lig_graph)
            lig_hid = self.run_linear(self.cfg.out_size, lig_hid)
        else:
            assert self.is_initialized()

        self.checkpoint()

        rec_hid = None
        if hasattr(x, "rec_graph"):
            rec_hid = self.get_hidden_feat(self.cfg.rec_encoder, x.rec_graph)
            if self.cfg.project_full_atom:
                full_cat_scal = self.make(CatScalEmbedding, self.cfg.full_atom_embed_size)
                full_ln = self.make_normalization()
                full_linear_out = self.make(LazyLinear, self.cfg.out_size)

                full_rec_hid = []
                tot_rec = 0
                rec_graph = x.rec_graph.dgl()
                for r, full_rec_data in zip(rec_graph.batch_num_nodes(), x.full_rec_data):
                    h1 = rec_hid[tot_rec + full_rec_data.res_index]
                    h2 = full_cat_scal(full_rec_data)
                    hid = torch.cat((h1, h2), -1)
                    hid = full_ln(hid)
                    hid = full_linear_out(F.leaky_relu(hid))
                    full_rec_hid.append(hid)
                    tot_rec += r
                rec_hid = full_rec_hid
            else:
                rec_hid = self.run_linear(self.cfg.out_size, rec_hid)

        else:
            assert self._initialized

        self.checkpoint()

        ret = []
        rbf_out = self.make(LazyLinear, self.cfg.rbf_steps)
        for i in range(len(x)):
            hid_cat = []
            if lig_hid is not None:
                hid_cat.append(lig_hid[x.lig_graph.node_slices[i]])
            if rec_hid is not None:
                if self.cfg.project_full_atom:
                    hid_cat.append(rec_hid[i])
                else:
                    hid_cat.append(rec_hid[x.rec_graph.node_slices[i]])
            all_hid = torch.cat(hid_cat, 0)
            op_mat = torch.einsum("xi,yj->xyij", all_hid, all_hid).reshape((all_hid.shape[0], all_hid.shape[0], -1))
            rbf_mat = rbf_out(op_mat)
            ret.append(rbf_mat)

        self.scale_output = self.make(ScaleOutput, self.cfg.energy_bias)

        return ret

    @staticmethod
    def get_energy_single(cfg, inter_mat, coords, weight, bias):
        dists = cdist_diff(coords, coords)
        rbfs = rbf_encode(dists, cfg.rbf_start,cfg.rbf_end,cfg.rbf_steps)
        # print(inter_mat.shape, dists.shape, rbfs.shape)
        inv_ident = (~torch.eye(dists.shape[-2], dtype=bool, device=coords.device).unsqueeze(-1))
        if rbfs.dim() == 4:
            assert inter_mat.dim() == 3
            inter_mat = inter_mat.unsqueeze(0)
            inv_ident = inv_ident.unsqueeze(0)
        interact = inter_mat*rbfs*cfg.energy_scale
        interact = interact*inv_ident
        return interact.mean((-3,-2,-1))*weight + bias

    def get_combined_coords(self, x, lig_pose):
        coord_cat = []
        if lig_pose is not None:
            coord_cat.append(lig_pose.coord)
        rec_coord = None
        if hasattr(x, "full_rec_data"):
            rec_coord = x.full_rec_data.coord
        elif hasattr(x, "rec_graph"):
            rec_coord = x.rec_graph.ndata.coord
        if rec_coord is not None:
            if lig_pose is not None and lig_pose.coord.dim() == 3:
                rec_coord = rec_coord.unsqueeze(0).repeat((lig_pose.coord.shape[0], 1, 1))
            coord_cat.append(rec_coord)
        return torch.cat(coord_cat, -2)

    def get_energy(self, inter_mats, x, lig_pose):
        bias, weight = self.scale_output.parameters()
        Us = []
        for inter, x0, pose in zip(inter_mats, x, lig_pose):
            coords = self.get_combined_coords(x0, pose)
            Us.append(SymForceField.get_energy_single(self.cfg, inter, coords, weight, bias))
        return torch.stack(Us)