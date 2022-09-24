import torch
import torch.nn as nn
import torch.nn.functional as F

from dgllife.model.gnn import MPNNGNN

from models.cat_scal_embedding import CatScalEmbedding

class LearnableFF(nn.Module):
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
        
    def forward(self, batch):
        rec_graph = batch.rec.dgl_batch
        lig_graph = batch.lig.dgl_batch
        rec_hid = self.rec_node_embed(batch.rec.ndata)
        lig_hid = self.lig_node_embed(batch.lig.ndata)
        rec_edge_feat = self.rec_edge_embed(batch.rec.edata)
        lig_edge_feat = self.lig_edge_embed(batch.lig.edata)

        batch_rec_feat = self.rec_gnn(rec_graph, rec_hid, rec_edge_feat)
        batch_lig_feat = self.lig_gnn(lig_graph, lig_hid, lig_edge_feat)

        tot_rec = 0
        tot_lig = 0
        Us = []
        for r, l in zip(rec_graph.batch_num_nodes(), lig_graph.batch_num_nodes()):
            lig_feat = batch_lig_feat[tot_lig:tot_lig+l]
            rec_feat = batch_rec_feat[tot_rec:tot_rec+r]

            init_lig_coord =  batch.lig.ndata.coord[tot_lig:tot_lig+l]
            rec_coord =  batch.rec.ndata.coord[tot_rec:tot_rec+r]

            # ensure the centroids of both lig and rec are at origin
            init_lig_coord = init_lig_coord - init_lig_coord.mean(0)
            rec_coord = rec_coord - rec_coord.mean(0)

            # initialize random rot and translation
            rot, _ = torch.linalg.qr(torch.randn((3,3), device=init_lig_coord.device))
            trans = torch.randn((3,), device=init_lig_coord.device) * self.cfg.model.trans_dist

            rot = nn.Parameter(rot)
            trans = nn.Parameter(trans)
            optim = torch.optim.AdamW([rot, trans], self.cfg.model.inner_optim_lr)

            #optimize rot and trans!
            for i in range(self.cfg.model.inner_optim_steps):
                # move ligand
                lig_coord = torch.einsum('ij,bj->bi', rot, init_lig_coord) + trans

                atn_coefs = torch.einsum('lf,rf->lr', lig_feat, rec_feat)
                dists = torch.cdist(lig_coord, rec_coord)

                U = (atn_coefs/(dists**2)).mean()
                U.backward(retain_graph=True)
                optim.step()
                optim.zero_grad()

            Us.append(U)

            tot_rec += r
            tot_lig += l

        # negate because affiinity is proportional to negative of energy
        return -torch.stack(Us)