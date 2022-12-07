import torch
import torch.nn as nn
import torch.nn.functional as F

from dgllife.model.gnn import MPNNGNN

from terrace.batch import Batch
from models.cat_scal_embedding import CatScalEmbedding
from datasets.data_types import PredData, EnergyPredData

class Diffusion(nn.Module):
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
                               self.cfg.model.out_size,
                               lig_cfg.edge_hidden_size,
                               lig_cfg.num_mpnn_layers)

        self.rec_gnn = MPNNGNN(self.rec_node_embed.total_dim,
                               self.rec_edge_embed.total_dim,
                               self.cfg.model.out_size,
                               rec_cfg.edge_hidden_size,
                               rec_cfg.num_mpnn_layers)

    def get_hidden_feat(self, batch):
        rec_graph = batch.rec.dgl_batch
        lig_graph = batch.lig.dgl_batch
        rec_hid = self.rec_node_embed(batch.rec.ndata)
        lig_hid = self.lig_node_embed(batch.lig.ndata)
        rec_edge_feat = self.rec_edge_embed(batch.rec.edata)
        lig_edge_feat = self.lig_edge_embed(batch.lig.edata)

        batch_rec_feat = self.rec_gnn(rec_graph, rec_hid, rec_edge_feat)
        batch_lig_feat = self.lig_gnn(lig_graph, lig_hid, lig_edge_feat)

        return batch_rec_feat, batch_lig_feat
        
    def get_energy(self,
                   batch,
                   batch_rec_feat,
                   batch_lig_feat,
                   batch_rot,
                   batch_trans):
        
        all_Us = []

        rec_graph = batch.rec.dgl_batch
        lig_graph = batch.lig.dgl_batch

        tot_rec = 0
        tot_lig = 0
        for i, (r, l) in enumerate(zip(rec_graph.batch_num_nodes(), lig_graph.batch_num_nodes())):
            
            rot = batch_rot # [i]
            trans = batch_trans #[i]
            
            lig_feat = batch_lig_feat[tot_lig:tot_lig+l]
            rec_feat = batch_rec_feat[tot_rec:tot_rec+r]

            lig_coord =  batch.lig.ndata.coord[tot_lig:tot_lig+l]
            rec_coord =  batch.rec.ndata.coord[tot_rec:tot_rec+r]

            tot_rec += r
            tot_lig += l

            # ensure the centroids of both lig and rec are at origin
            lig_coord = lig_coord - lig_coord.mean(0)
            rec_coord = rec_coord - rec_coord.mean(0)

            # move ligand
            new_lig_coord = torch.einsum('nij,bj->nbi', rot, lig_coord)  + trans.unsqueeze(1)

            atn_coefs = torch.einsum('lf,rf->lr', lig_feat, rec_feat)

            # cdist is the simplest way to compute dist matrix
            # dists = torch.cdist(new_lig_coord, rec_coord)
            
            # non-cdist vectorized way
            Us = []
            for i in range(trans.size(0)):
                lc_ex = new_lig_coord[i].unsqueeze(1).expand(-1,rec_coord.size(0),-1)
                rc_ex = rec_coord.unsqueeze(0).expand(new_lig_coord[i].size(0),-1,-1)
                dists = torch.sqrt(((lc_ex - rc_ex)**2).sum(-1))

                U = (atn_coefs/(dists)).sum()
                Us.append(U)

            all_Us.append(torch.stack(Us))

        return torch.stack(all_Us)

    def energy_grad(self,
                    batch,
                    batch_rec_feat,
                    batch_lig_feat,
                    pre_rot,
                    trans):

        with torch.set_grad_enabled(True):
            pre_rot.requires_grad_()
            trans.requires_grad_()

            rot, _ = torch.linalg.qr(pre_rot)
            U = self.get_energy(batch,
                                batch_rec_feat,
                                batch_lig_feat,
                                rot,
                                trans)

            U_mean = U.mean()

            pre_rot_grad, trans_grad = torch.autograd.grad(U_mean, [pre_rot, trans], create_graph=True)
        return pre_rot_grad, trans_grad

    def get_diffused_transforms(self, device, timesteps=None):
        diff_cfg = self.cfg.model.diffusion
        if timesteps is None:
            timesteps = diff_cfg.timesteps

        rot_sigma = torch.linspace(0.0, diff_cfg.max_rot_sigma, timesteps, device=device)
        trans_sigma = torch.linspace(0.0, diff_cfg.max_trans_sigma, timesteps, device=device)
        
        # a bunch of identity matrices
        pre_rot = torch.eye(3, device=device).unsqueeze(0).repeat(diff_cfg.timesteps,1,1)
        # add noise
        pre_rot += torch.randn((timesteps,3,3), device=device)*rot_sigma.view((-1,1,1))
        
        trans = torch.randn((timesteps,3), device=device)*trans_sigma.view((-1,1))

        return pre_rot, trans

    def pred_pose(self,
                  batch,
                  batch_rec_feat,
                  batch_lig_feat,
                  pre_rot,
                  trans):

        rot_grad, trans_grad = self.energy_grad(batch,
                                                batch_rec_feat,
                                                batch_lig_feat,
                                                pre_rot,
                                                trans)

        final_rot, _ = torch.linalg.qr(pre_rot - self.cfg.model.grad_coef*rot_grad)
        final_trans = trans - self.cfg.model.grad_coef*trans_grad

        return final_rot, final_trans
            

    def diffuse(self, batch):

        batch_rec_feat, batch_lig_feat = self.get_hidden_feat(batch)
        device = batch_rec_feat.device

        pre_rot, trans = self.get_diffused_transforms(device)

        return self.pred_pose(batch, 
                              batch_rec_feat,
                              batch_lig_feat,
                              pre_rot,
                              trans)

    def apply_transformation(self, batch, rot, trans):

        rec_graph = batch.rec.dgl_batch
        lig_graph = batch.lig.dgl_batch

        ret = []

        tot_rec = 0
        tot_lig = 0
        for i, (r, l) in enumerate(zip(rec_graph.batch_num_nodes(), lig_graph.batch_num_nodes())):

            lig_coord =  batch.lig.ndata.coord[tot_lig:tot_lig+l]
            rec_coord =  batch.rec.ndata.coord[tot_rec:tot_rec+r]

            tot_rec += r
            tot_lig += l

            # ensure the centroids of both lig and rec are at origin
            lig_coord = lig_coord - lig_coord.mean(0)
            rec_coord = rec_coord - rec_coord.mean(0)

            # move ligand
            new_lig_coord = torch.einsum('nij,bj->nbi', rot, lig_coord)  + trans.unsqueeze(1)
            ret.append(new_lig_coord)

        return ret

    def infer(self, batch):
        """ Final inference -- predict lig_coords directly after randomizing """

        batch_rec_feat, batch_lig_feat = self.get_hidden_feat(batch)
        device = batch_rec_feat.device

        pre_rot, trans = self.get_diffused_transforms(device)
        pre_rot = pre_rot[-2:-1]
        trans = trans[-2:-1]

        for t in range(self.cfg.model.diffusion.timesteps):
            pre_rot, trans = self.pred_pose(batch, 
                                            batch_rec_feat,
                                            batch_lig_feat,
                                            pre_rot,
                                            trans)

        ret = self.apply_transformation(batch, pre_rot, trans)
        return [ coords[0] for coords in ret ]

    def forward(self, batch):
        return self.diffuse(batch)