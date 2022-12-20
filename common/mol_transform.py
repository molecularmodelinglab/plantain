from dataclasses import dataclass
from typing import Optional
import roma
import torch

from terrace.batch import Batchable

# todo: this class encompasses both batch of moltransform and moltransform
# make this into batchable somehow while preserving methods
@dataclass
class MolTransform:

    # todo: torsional angles!
    rot: torch.Tensor
    trans: torch.Tensor
    rot_frac: Optional[torch.Tensor] = None
    trans_sigma: Optional[torch.Tensor] = None

    @staticmethod
    def make_diffused(timesteps, diff_cfg, batch_size, device):
        trans_sigma = torch.linspace(0.0, diff_cfg.max_trans_sigma, timesteps, device=device).view((1,-1,1))
        trans = torch.randn((batch_size,timesteps,3), device=device)*trans_sigma
        
        rand_rot = torch.stack([torch.stack([roma.random_rotvec(device=device) for t in range(timesteps)]) for b in range(batch_size)])
        rot_frac = torch.linspace(0.0, diff_cfg.max_rot_frac, timesteps, device=device).view(1,-1,1)
        rot = rand_rot*rot_frac
        return MolTransform(rot, trans, rot_frac, trans_sigma)

    def __getitem__(self, i):
        print(self.rot_frac)
        return MolTransform(self.rot[i], self.trans[i], self.rot_frac[i], self.trans_sigma[i])

    def apply(self, coord):
        rot_mat = roma.rotvec_to_rotmat(self.rot)
        return torch.einsum('nij,bj->nbi', rot_mat, coord)  + self.trans.unsqueeze(1)

    def apply_to_graph_batch(self, batch):
        ret = []
        tot_lig = 0
        for i, l in enumerate(batch.dgl_batch.batch_num_nodes()):

            transform = self[i]
            coord = batch.ndata.coord[tot_lig:tot_lig+l]

            tot_lig += l

            # move ligand
            new_coord = transform.apply(coord)
            ret.append(new_coord)
        return ret

    def grad(self, U):
        rot_grad, trans_grad = torch.autograd.grad(U, [self.rot, self.trans], create_graph=True)
        return MolTransform(rot_grad, trans_grad)

    def requires_grad(self):
        self.rot.requires_grad_()
        self.trans.requires_grad_()

    def update_from_grad(self, grad):
        # todo: should we square the rot frac? I don't know math
        # rot_frac_sq = (self.rot_frac**2).view((1,-1,1))
        trans_sigma_sq = (self.trans_sigma**2)
        # print(trans_sigma_sq)
        rot = self.rot - grad.rot*0.1#(self.rot_frac)#*trans_sigma_sq
        trans = self.trans - grad.trans*trans_sigma_sq
        return MolTransform(rot, trans, self.rot_frac, self.trans_sigma)