import torch
import torch.nn as nn
import torch.nn.functional as F
from common.pose_transform import MultiPose, Pose
from models.attention_gnn import MPNN
from terrace import Module
from terrace.batch import Batch
from terrace.dataframe import DFRow
from terrace.module import LazyLayerNorm, LazyLinear, LazyMultiheadAttention
from .model import ClassifyActivityModel
from .cat_scal_embedding import CatScalEmbedding
from .graph_embedding import GraphEmbedding
from .force_field import ScaleOutput, rbf_encode, cdist_diff

class Twister(Module, ClassifyActivityModel):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.model

    @staticmethod
    def get_name():
        return "twister"

    def get_tasks(self):
        return [ "score_activity_class",
                 "classify_activity",
                 "score_pose",
                 "predict_activity",
                 # "score_activity_regr" 
                ]

    def get_input_feats(self):
        ret = [ "lig_graph", "rec_graph", "lig_docked_poses" ]
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

    def forward(self, x, lig_poses=None):
        self.start_forward()
        
        if lig_poses is None:
            lig_poses = x.lig_docked_poses

        inter_mat = self.get_initial_interaction_matrix(x)
        flat_output = self.get_outputs_from_inter_mat(inter_mat, 3).unsqueeze(1)

        if x.lig_docked_poses[0].coord.shape[0] > 0:
            dist_inter_mat = self.get_dist_inter_mat(x, inter_mat, lig_poses)
            dist_output = self.get_outputs_from_inter_mat(dist_inter_mat, 2)
            pose_scores = dist_output[:,:,1]
        else:
            dist_output = torch.zeros((len(x), 0, 3), device=flat_output.device)
            pose_scores =torch.zeros((len(x), 0),  device=flat_output.device)

        full_output = torch.cat((flat_output, dist_output), 1)
        preds = full_output[:,:,0]
        atn = full_output[:,:,1]
        act_preds = full_output[:,:,2]
        
        atn = torch.softmax(atn, -1)
        score = (preds*atn).sum(-1)
        act = (act_preds*atn).sum(-1)

        full_pose_scores = full_output[:,:,1]

        return Batch(DFRow,
                     score=score,
                     activity=act,
                     activity_score=act,
                     pose_scores=pose_scores,
                     full_pose_scores=full_pose_scores,
                     full_activity_scores=preds,
                     full_activities=act_preds)

    def predict_train(self, x, y, task_names, split, batch_idx):
        if hasattr(y, "lig_crystal_pose"):
            lig_poses = MultiPose(coord=[coord.unsqueeze(0) for coord in y.lig_crystal_pose.coord])
        else:
            lig_poses = x.lig_docked_poses
        return self.finalize_prediction(x, self(x, lig_poses), task_names)

    def get_initial_interaction_matrix(self, x):
        """ Returns tensor of shape L x R x H, for use in further no-pose and pose classification """

        lig_hid = self.get_hidden_feat(self.cfg.lig_encoder, x.lig_graph)
        lig_hid = self.run_linear(self.cfg.single_out_size, lig_hid)


        rec_hid = self.get_hidden_feat(self.cfg.rec_encoder, x.rec_graph)
        if self.cfg.project_full_atom:
            full_cat_scal = self.make(CatScalEmbedding, self.cfg.full_atom_embed_size)
            full_ln = self.make_normalization()
            full_linear_out = self.make(LazyLinear, self.cfg.single_out_size)

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

        ret = []
        # inter_ln = self.make_normalization()
        inter_linear_out = self.make(LazyLinear, self.cfg.interact_out_size)
        # out_ln = self.make_normalization()
        for i in range(len(x)):
            lf = lig_hid[x.lig_graph.node_slices[i]]
            rf = rec_hid[i]
            op_mat = torch.einsum("xi,yj->xyij", lf, rf).reshape((lf.shape[0], rf.shape[0], -1))
            # inter_hid = inter_ln(op_mat)
            inter_hid = op_mat
            inter_hid = inter_linear_out(F.leaky_relu(inter_hid))
            # inter_hid = out_ln(inter_hid)
            ret.append(inter_hid)

        return ret

    def get_outputs_from_inter_mat(self, inter_mats, num_outputs):
        out = []
        atn_linear = self.make(LazyLinear, 1)
        out_mat_linear = self.make(LazyLinear, self.cfg.interact_out_size)
        for mat in inter_mats:
            atn = atn_linear(mat)
            out_mat = out_mat_linear(mat)
            if mat.dim() == 4:
                atn = F.softmax(atn.reshape((atn.shape[0], -1)),1).reshape(*atn.shape)
                out.append((atn*out_mat).sum(1).sum(1))
            else:
                atn = F.softmax(atn.reshape(-1),0).reshape(*atn.shape)
                out.append((atn*out_mat).sum(0).sum(0))
        out = torch.stack(out)
        return self.run_linear(num_outputs, out.reshape((-1, out.shape[-1]))).reshape((*out.shape[:-1], -1))

    def get_dist_inter_mat(self, x, inter_mats, lig_poses):
        # in_ln = self.make_normalization()
        linear_in = self.make(LazyLinear, self.cfg.interact_dist_in_size)
        # out_ln = self.make_normalization()
        linear_out = self.make(LazyLinear, self.cfg.interact_hid_size)
        ret = []
        for mat, rec_coord, lig_coord in zip(inter_mats, x.full_rec_data.coord, lig_poses.coord):
            dist = cdist_diff(lig_coord, rec_coord)
            rbfs = rbf_encode(dist, self.cfg.rbf_start,self.cfg.rbf_end,self.cfg.rbf_steps)
            in_mat = linear_in(F.leaky_relu(mat))
            combo_mat = torch.einsum("...i,...j->...ij", in_mat, rbfs).reshape(*rbfs.shape[:-1], -1)
            # combo_mat = in_ln(combo_mat)
            out_mat = linear_out(F.leaky_relu(combo_mat))
            # out_mat = out_ln(out_mat)
            ret.append(out_mat)
        return ret