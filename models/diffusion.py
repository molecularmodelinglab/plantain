import torch
import torch.nn as nn
import torch.nn.functional as F

from dgllife.model.gnn import MPNNGNN

from common.mol_transform import MolTransform
from terrace.batch import Batch
from models.cat_scal_embedding import CatScalEmbedding
from datasets.data_types import PredData, EnergyPredData

def rbf_encode(dists, start, end, steps):
    mu = torch.linspace(start, end, steps, device=dists.device)
    sigma = (start - end)/steps
    dists_expanded = dists.unsqueeze(-1).repeat(1,1,mu.size(0))
    mu_expanded = mu.view(1,1,-1)
    diff = ((dists_expanded - mu_expanded)/sigma)**2
    return torch.exp(-diff)

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
                               self.cfg.model.out_size*self.cfg.model.rbf_steps,
                               lig_cfg.edge_hidden_size,
                               lig_cfg.num_mpnn_layers)

        self.rec_gnn = MPNNGNN(self.rec_node_embed.total_dim,
                               self.rec_edge_embed.total_dim,
                               self.cfg.model.out_size*self.cfg.model.rbf_steps,
                               rec_cfg.edge_hidden_size,
                               rec_cfg.num_mpnn_layers)

    def get_hidden_feat(self, batch):
        rec_graph = batch.rec.dgl_batch
        lig_graph = batch.lig.dgl_batch
        rec_hid = self.rec_node_embed(batch.rec.ndata)
        lig_hid = self.lig_node_embed(batch.lig.ndata)
        rec_edge_feat = self.rec_edge_embed(batch.rec.edata)
        lig_edge_feat = self.lig_edge_embed(batch.lig.edata)

        batch_rec_feat = self.rec_gnn(rec_graph, rec_hid, rec_edge_feat).view(-1, self.cfg.model.rbf_steps, self.cfg.model.out_size)
        batch_lig_feat = self.lig_gnn(lig_graph, lig_hid, lig_edge_feat).view(-1, self.cfg.model.rbf_steps, self.cfg.model.out_size)

        return batch_rec_feat, batch_lig_feat
        
    def get_energy(self,
                   batch,
                   batch_rec_feat,
                   batch_lig_feat,
                   batch_transform):
        
        all_Us = []

        rec_graph = batch.rec.dgl_batch
        lig_graph = batch.lig.dgl_batch

        tot_rec = 0
        tot_lig = 0
        for i, (r, l) in enumerate(zip(rec_graph.batch_num_nodes(), lig_graph.batch_num_nodes())):
            
            transform = batch_transform[i]
            
            lig_feat = batch_lig_feat[tot_lig:tot_lig+l]
            rec_feat = batch_rec_feat[tot_rec:tot_rec+r]

            lig_coord =  batch.lig.ndata.coord[tot_lig:tot_lig+l]
            rec_coord =  batch.rec.ndata.coord[tot_rec:tot_rec+r]

            tot_rec += r
            tot_lig += l

            # move ligand
            new_lig_coord = transform.apply(lig_coord)

            atn_coefs = torch.einsum('lef,ref->lre', lig_feat, rec_feat)

            # cdist is the simplest way to compute dist matrix
            # dists = torch.cdist(new_lig_coord, rec_coord)
            
            # non-cdist vectorized way
            Us = []
            for i in range(transform.trans.size(0)):
                lc_ex = new_lig_coord[i].unsqueeze(1).expand(-1,rec_coord.size(0),-1)
                rc_ex = rec_coord.unsqueeze(0).expand(new_lig_coord[i].size(0),-1,-1)
                dists = torch.sqrt(((lc_ex - rc_ex)**2).sum(-1))
                rbfs = rbf_encode(dists, self.cfg.model.rbf_start,self.cfg.model.rbf_end,self.cfg.model.rbf_steps,)

                U = (atn_coefs*rbfs*self.cfg.model.energy_scale).sum()
                Us.append(U)

            all_Us.append(torch.stack(Us))

        return torch.stack(all_Us)

    def energy_grad(self,
                    batch,
                    batch_rec_feat,
                    batch_lig_feat,
                    transform):

        with torch.set_grad_enabled(True):
            transform.requires_grad()
            U = self.get_energy(batch,
                                batch_rec_feat,
                                batch_lig_feat,
                                transform)
            U_sum = U.sum()

            return transform.grad(U_sum) 

    def get_diffused_transforms(self, batch_size, device, timesteps=None):
        diff_cfg = self.cfg.model.diffusion
        if timesteps is None:
            timesteps = diff_cfg.timesteps

        return MolTransform.make_diffused(timesteps, diff_cfg, batch_size, device)

    def pred_pose(self,
                  batch,
                  batch_rec_feat,
                  batch_lig_feat,
                  transform):

        grad = self.energy_grad(batch,
                                batch_rec_feat,
                                batch_lig_feat,
                                transform)
        return transform.update_from_grad(grad)
        
            

    def get_diffused_coords(self, batch):
        
        batch_rec_feat, batch_lig_feat = self.get_hidden_feat(batch)
        device = batch_rec_feat.device

        transform = self.get_diffused_transforms(len(batch), device)

        return self.apply_transformation(batch, transform)

    def diffuse(self, batch):

        batch_rec_feat, batch_lig_feat = self.get_hidden_feat(batch)
        device = batch_rec_feat.device

        transform = self.get_diffused_transforms(len(batch), device)

        return self.pred_pose(batch, 
                              batch_rec_feat,
                              batch_lig_feat,
                              transform)

    def apply_transformation(self, batch, batch_transform):

        rec_graph = batch.rec.dgl_batch
        lig_graph = batch.lig.dgl_batch

        ret = []

        tot_rec = 0
        tot_lig = 0
        for i, (r, l) in enumerate(zip(rec_graph.batch_num_nodes(), lig_graph.batch_num_nodes())):

            transform = batch_transform[i]

            lig_coord =  batch.lig.ndata.coord[tot_lig:tot_lig+l]

            tot_rec += r
            tot_lig += l

            # move ligand
            new_lig_coord = transform.apply(lig_coord)
            ret.append(new_lig_coord)

        return ret

    def infer(self, batch, ret_all_coords=False):
        """ Final inference -- predict lig_coords directly after randomizing """

        batch_rec_feat, batch_lig_feat = self.get_hidden_feat(batch)
        device = batch_rec_feat.device

        transform = self.get_diffused_transforms(len(batch), device)[:,-2:-1]

        all_coords = []
        for t in range(self.cfg.model.diffusion.timesteps):
            transform = self.pred_pose(batch, 
                                       batch_rec_feat,
                                       batch_lig_feat,
                                       transform)
            if ret_all_coords:
                all_coords.append(self.apply_transformation(batch, transform))

        if ret_all_coords:
            ret = [ [] for i in range(len(batch)) ]
            for coords in all_coords:
                for i, c in enumerate(coords):
                    ret[i].append(c[0])
            return ret
        
        ret = self.apply_transformation(batch, transform)
        return [ coords[0] for coords in ret ]

    def forward(self, batch):
        return self.diffuse(batch)