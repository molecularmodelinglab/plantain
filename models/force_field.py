import torch
import torch.nn as nn

from dgllife.model.gnn import MPNNGNN
from models.cat_scal_embedding import CatScalEmbedding

def rbf_encode(dists, start, end, steps):
    mu = torch.linspace(start, end, steps, device=dists.device)
    sigma = (start - end)/steps
    dists_expanded = dists.unsqueeze(-1).repeat(1,1,mu.size(0))
    mu_expanded = mu.view(1,1,-1)
    diff = ((dists_expanded - mu_expanded)/sigma)**2
    return torch.exp(-diff)

class ForceField(nn.Module):

    def __init__(self, cfg, in_node):
        super().__init__()
        self.cfg = cfg

        rec_cfg = cfg.model.rec_encoder
        lig_cfg = cfg.model.lig_encoder

        self.rec_node_embed = CatScalEmbedding(cfg.model.rec_encoder.node_embed_size,
                                               in_node.out_type_data.rec.ndata)

        self.lig_node_embed = CatScalEmbedding(cfg.model.lig_encoder.node_embed_size,
                                               in_node.out_type_data.ligs[0].ndata)

        self.rec_edge_embed = CatScalEmbedding(cfg.model.rec_encoder.edge_embed_size,
                                               in_node.out_type_data.rec.edata)

        self.lig_edge_embed = CatScalEmbedding(cfg.model.lig_encoder.edge_embed_size,
                                               in_node.out_type_data.ligs[0].edata)

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

    def get_hidden_feat(self, batch, conf_id):
        rec_graph = batch.rec.dgl_batch
        lig_graph = batch.ligs[conf_id].dgl_batch
        rec_hid = self.rec_node_embed(batch.rec.ndata)
        lig_hid = self.lig_node_embed(batch.ligs[conf_id].ndata)
        rec_edge_feat = self.rec_edge_embed(batch.rec.edata)
        lig_edge_feat = self.lig_edge_embed(batch.ligs[conf_id].edata)

        batch_rec_feat = self.rec_gnn(rec_graph, rec_hid, rec_edge_feat).view(-1, self.cfg.model.rbf_steps, self.cfg.model.out_size)
        batch_lig_feat = self.lig_gnn(lig_graph, lig_hid, lig_edge_feat).view(-1, self.cfg.model.rbf_steps, self.cfg.model.out_size)

        return batch_rec_feat, batch_lig_feat
        
    def get_energy(self,
                   batch,
                   batch_rec_feat,
                   batch_lig_feat,
                   conf_id):
        
        all_Us = []

        rec_graph = batch.rec.dgl_batch
        lig_graph = batch.ligs[conf_id].dgl_batch

        tot_rec = 0
        tot_lig = 0
        for i, (r, l) in enumerate(zip(rec_graph.batch_num_nodes(), lig_graph.batch_num_nodes())):
            
            lig_feat = batch_lig_feat[tot_lig:tot_lig+l]
            rec_feat = batch_rec_feat[tot_rec:tot_rec+r]

            lig_coord =  batch.ligs[conf_id].ndata.coord[tot_lig:tot_lig+l]
            rec_coord =  batch.rec.ndata.coord[tot_rec:tot_rec+r]

            tot_rec += r
            tot_lig += l

            atn_coefs = torch.einsum('lef,ref->lre', lig_feat, rec_feat)

            # cdist is the simplest way to compute dist matrix
            # dists = torch.cdist(new_lig_coord, rec_coord)
            
            # non-cdist vectorized way
            lc_ex = lig_coord.unsqueeze(1).expand(-1,rec_coord.size(0),-1)
            rc_ex = rec_coord.unsqueeze(0).expand(lig_coord.size(0),-1,-1)
            dists = torch.sqrt(((lc_ex - rc_ex)**2).sum(-1))
            rbfs = rbf_encode(dists, self.cfg.model.rbf_start,self.cfg.model.rbf_end,self.cfg.model.rbf_steps,)

            U = (atn_coefs*rbfs*self.cfg.model.energy_scale).sum()
            all_Us.append(U)

        return torch.stack(all_Us)

    # todo: add proper attention stuff if this shows promise
    def get_conformer_scores(self, batch):
        batch_rec_feat, batch_lig_feat = self.get_hidden_feat(batch, 0)
        ret = []
        for conf_id in range(len(batch.ligs)):
            ret.append(self.get_energy(batch, batch_rec_feat, batch_lig_feat, conf_id))
        return torch.stack(ret).T.unsqueeze(-1)

    def forward(self, batch, conf_id=0):
        batch_rec_feat, batch_lig_feat = self.get_hidden_feat(batch, conf_id)
        return self.get_energy(batch, batch_rec_feat, batch_lig_feat, conf_id)