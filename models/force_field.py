import torch
import torch.nn as nn
import torch.nn.functional as F
from common.pose_transform import Pose
from models.attention_gnn import MPNN, batched_feats, single_batched_feat
from terrace import Module
from terrace.batch import Batch
from terrace.dataframe import DFRow
from terrace.module import LazyLayerNorm, LazyLinear, LazyMultiheadAttention
from .model import ClassifyActivityModel
from .cat_scal_embedding import CatScalEmbedding
from .graph_embedding import GraphEmbedding

class ScaleOutput(nn.Module):

    def __init__(self, init_bias):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(init_bias, dtype=torch.float32))
        self.weight = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self, x):
        return x*self.weight + self.bias

def rbf_encode(dists, start, end, steps):
    mu = torch.linspace(start, end, steps, device=dists.device)
    sigma = (start - end)/steps
    dists_expanded = dists.unsqueeze(-1)# .repeat(1,1,mu.size(0))
    mu_expanded = mu.view(1,1,-1)
    diff = ((dists_expanded - mu_expanded)/sigma)**2
    return torch.exp(-diff)

def cdist_diff(x, y):
    """ cdist, but you can differentiate its derivative """
    x_ex = x.unsqueeze(-2)
    y_ex = y.unsqueeze(-3)
    # eps is needed cus otherwise we get nans in derivatives
    eps = 1e-10
    return torch.sqrt(((x_ex - y_ex)**2 + eps).sum(-1))

class ForceField(Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.model

    def get_input_feats(self):
        ret = [ "lig_graph", "rec_graph" ]
        if self.cfg.get("project_full_atom", False):
            ret.append("full_rec_data")
        return ret

    @staticmethod
    def energy_from_atn_coef(cfg, atn_coefs, coord1, coord2, kill_diag = False):
        # dists = torch.cdist(coord1, coord2)
        dists = cdist_diff(coord1, coord2)
        rbfs = rbf_encode(dists, cfg.rbf_start,cfg.rbf_end,cfg.rbf_steps)
        interact = atn_coefs*rbfs*cfg.energy_scale
        if kill_diag:
            interact = interact*(~torch.eye(interact.shape[0], dtype=bool, device=coord1.device).unsqueeze(-1))
            # interact = interact.clone()
            # interact.diagonal(dim1=0, dim2=1).zero_()
        return interact.sum()

    def get_hidden_feat(self, x):
        self.start_forward()

        lig_cfg = self.cfg.lig_encoder
        rec_cfg = self.cfg.rec_encoder

        lig_node_feats, lig_edge_feats = self.make(GraphEmbedding, self.cfg.lig_encoder)(x.lig_graph)
        rec_node_feats, rec_edge_feats = self.make(GraphEmbedding, self.cfg.rec_encoder)(x.rec_graph)

        lig_hid = self.make(LazyLinear, lig_cfg.node_out_size)(F.leaky_relu(lig_node_feats))
        rec_hid = self.make(LazyLinear, rec_cfg.node_out_size)(F.leaky_relu(rec_node_feats))

        if self.cfg.get("use_layer_norm", False):
            rec_hid = self.make(LazyLayerNorm)(rec_hid)
            lig_hid = self.make(LazyLayerNorm)(lig_hid)

        prev_lig_hid = []
        prev_rec_hid = []

        for layer in range(self.cfg.num_mpnn_layers):
            lig_hid = self.make(MPNN)(x.lig_graph, F.leaky_relu(lig_hid), lig_edge_feats)
            rec_hid = self.make(MPNN)(x.rec_graph, F.leaky_relu(rec_hid), rec_edge_feats)

            if self.cfg.get("use_layer_norm", False):
                rec_hid = self.make(LazyLayerNorm)(rec_hid)
                lig_hid = self.make(LazyLayerNorm)(lig_hid)

            if self.cfg.get("inner_attention", False):
                rec_hid = (rec_hid + self.make(LazyMultiheadAttention, 1)(rec_hid, lig_hid, lig_hid)[0])
                lig_hid = (lig_hid + self.make(LazyMultiheadAttention, 1)(lig_hid, rec_hid, rec_hid)[0])
                if self.cfg.get("use_layer_norm", False):
                    rec_hid = self.make(LazyLayerNorm)(rec_hid)
                    lig_hid = self.make(LazyLayerNorm)(lig_hid)

            # make it residual!
            prev_layer = layer - 2
            if prev_layer >= 0:
                lig_hid = lig_hid + prev_lig_hid[prev_layer]
                rec_hid = rec_hid + prev_rec_hid[prev_layer]

            prev_lig_hid.append(lig_hid)
            prev_rec_hid.append(rec_hid)

        if self.cfg.get("null_pose_option", False):
            aux_lig_hid = self.make(LazyLinear, self.cfg.null_pose_out_size)(F.leaky_relu(lig_hid))

        use_intra_lig = self.cfg.get("intra_lig_energy", False)
        if use_intra_lig:
            lig_hid = self.make(LazyLinear, self.cfg.out_size*self.cfg.rbf_steps*2)(F.leaky_relu(lig_hid))
            lig_hid = lig_hid.view(-1, 2, self.cfg.rbf_steps, self.cfg.out_size)
        else:
            lig_hid = self.make(LazyLinear, self.cfg.out_size*self.cfg.rbf_steps)(F.leaky_relu(lig_hid))
            lig_hid = lig_hid.view(-1, self.cfg.rbf_steps, self.cfg.out_size)

        if self.cfg.get("use_layer_norm", False):
            rec_hid = self.make(LazyLayerNorm)(rec_hid)
            lig_hid = self.make(LazyLayerNorm)(lig_hid)

        if self.cfg.get("project_full_atom", False):
            full_cat_scal = self.make(CatScalEmbedding, self.cfg.full_atom_embed_size)
            if self.cfg.get("use_layer_norm", False):
                full_ln = self.make(LazyLayerNorm)
            full_linear_out = self.make(LazyLinear, self.cfg.out_size*self.cfg.rbf_steps)
            if self.cfg.get("null_pose_option", False):
                aux_linear_out = self.make(LazyLinear, self.cfg.null_pose_out_size)

            full_rec_hid = []
            aux_rec_hid = []
            tot_rec = 0
            rec_graph = x.rec_graph.dgl()
            for r, full_rec_data in zip(rec_graph.batch_num_nodes(), x.full_rec_data):
                h1 = rec_hid[tot_rec + full_rec_data.res_index]
                h2 = full_cat_scal(full_rec_data)
                hid = torch.cat((h1, h2), -1)
                if self.cfg.get("use_layer_norm", False):
                    hid = full_ln(hid)

                if self.cfg.get("null_pose_option", False):
                    aux_hid = aux_linear_out(F.leaky_relu(hid))
                    aux_rec_hid.append(aux_hid)
                
                hid = full_linear_out(F.leaky_relu(hid))
                hid = hid.view(-1, self.cfg.rbf_steps, self.cfg.out_size)
                full_rec_hid.append(hid)
                tot_rec += r
            rec_hid = full_rec_hid
        else:
            assert self.cfg.get("null_pose_option", False)
            rec_hid = self.make(LazyLinear, self.cfg.out_size*self.cfg.rbf_steps)(F.leaky_relu(rec_hid))
            rec_hid = rec_hid.view(-1, self.cfg.rbf_steps, self.cfg.out_size)

        aux_energy = []
        if self.cfg.get("null_pose_option", False):
            for lig_feat, rec_feat in zip(single_batched_feat(x, aux_lig_hid), aux_rec_hid):
                op = torch.einsum('lf,rf->lr', lig_feat, rec_feat)
                aux_energy.append(op.mean())
            return rec_hid, lig_hid, torch.stack(aux_energy)

        if "energy_bias" in self.cfg:
            self.scale_output = self.make(ScaleOutput, self.cfg.energy_bias)
        return rec_hid, lig_hid
        
    
    @staticmethod
    def get_energy_single(cfg,
                          rec_feat,
                          lig_feat,
                          rec_coord,
                          lig_coord,
                          weight,
                          bias):
        use_intra_lig = cfg.get("intra_lig_energy", False)
        if use_intra_lig:
            atn_coefs = torch.einsum('lef,ref->lre', lig_feat[:,0], rec_feat)
            if cfg.get("asym_lig_energy", False):
                ll_atn = torch.einsum("lef,ref->lre", lig_feat[:,0], lig_feat[:,1])
            else:
                ll_atn = torch.einsum("lef,ref->lre", lig_feat[:,1], lig_feat[:,1])
        else:
            atn_coefs = torch.einsum('lef,ref->lre', lig_feat, rec_feat)

        U = ForceField.energy_from_atn_coef(cfg, atn_coefs, lig_coord, rec_coord)
        if use_intra_lig:
            U += ForceField.energy_from_atn_coef(cfg, ll_atn, lig_coord, lig_coord, kill_diag=True)
        return bias + U*weight


    def get_energy(self,
                   batch,
                   batch_rec_feat,
                   batch_lig_feat,
                   batch_lig_poses):
        
        all_Us = []

        rec_graph = batch.rec_graph.dgl()
        lig_graph = batch.lig_graph.dgl()

        tot_rec = 0
        tot_lig = 0
        for i, (r, l) in enumerate(zip(rec_graph.batch_num_nodes(), lig_graph.batch_num_nodes())):
            
            lig_feat = batch_lig_feat[tot_lig:tot_lig+l]
            if self.cfg.get("project_full_atom", False):
                rec_feat = batch_rec_feat[i]
                rec_coord = batch.full_rec_data.coord[i]
            else:
                rec_feat = batch_rec_feat[tot_rec:tot_rec+r]
                rec_coord =  batch.rec_graph.ndata.coord[tot_rec:tot_rec+r]

            tot_rec += r
            tot_lig += l

            use_intra_lig = self.cfg.get("intra_lig_energy", False)
            if use_intra_lig:
                atn_coefs = torch.einsum('lef,ref->lre', lig_feat[:,0], rec_feat)
                if self.cfg.get("asym_lig_energy", False):
                    ll_atn = torch.einsum("lef,ref->lre", lig_feat[:,0], lig_feat[:,1])
                else:
                    ll_atn = torch.einsum("lef,ref->lre", lig_feat[:,1], lig_feat[:,1])
            else:
                atn_coefs = torch.einsum('lef,ref->lre', lig_feat, rec_feat)

            def get_U(lig_coord):
                U = ForceField.energy_from_atn_coef(self.cfg, atn_coefs, lig_coord, rec_coord)
                if use_intra_lig:
                    U += ForceField.energy_from_atn_coef(self.cfg, ll_atn, lig_coord, lig_coord, kill_diag=True)
                return U

            # very hacky way of allowing diffusion model to pass multiple transforms
            # to the energy function
            if len(batch_lig_poses[i].coord.shape) == 3:
                # print(batch_lig_poses[i].coord.shape)
                Us = []
                for lig_coord in batch_lig_poses.coord[i]:
                    Us.append(get_U(lig_coord))
                all_Us.append(torch.stack(Us))
            else:
                assert len(batch_lig_poses[i].coord.shape) == 2
                lig_coord =  batch_lig_poses[i].coord
                all_Us.append(get_U(lig_coord))

        ret = torch.stack(all_Us)
        if "energy_bias" in self.cfg:
            ret = self.scale_output(ret)
        return ret

class ForceFieldClassifier(Module, ClassifyActivityModel):
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.model
        self.force_field = ForceField(cfg)

    @staticmethod
    def get_name() -> str:
        return "force_field"

    def get_input_feats(self):
        return ["lig_docked_poses"] + self.force_field.get_input_feats()

    def forward(self, batch):
        self.start_forward()

        hid_feat = self.force_field.get_hidden_feat(batch)
        if self.cfg.get("null_pose_option", False):
            batch_rec_feat, batch_lig_feat, U_null = hid_feat
        else:
            batch_rec_feat, batch_lig_feat = hid_feat

        ret = []
        for conf_id in range(len(batch.lig_docked_poses.coord[0])):
            pose = Batch(Pose, coord=[coord[conf_id] for coord in batch.lig_docked_poses.coord])
            ret.append(self.force_field.get_energy(batch, batch_rec_feat, batch_lig_feat, pose))
        if self.cfg.get("null_pose_option", False):
            ret.append(U_null)

        U = torch.stack(ret).T

        if self.cfg.get("multi_pose_attention", False):
            lin_out = self.make(LazyLinear, 2)(U.unsqueeze(-1))
            pose_scores = lin_out[...,0]
            atn = lin_out[...,1]
            atn = torch.softmax(atn[:,:-1], -1)
            score = (pose_scores[:,:-1]*atn).sum(-1)
        else:
            score = U[:,0]
            pose_scores = U

        return Batch(DFRow, score=score, pose_scores=pose_scores)
