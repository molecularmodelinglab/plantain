import roma
import torch
from rdkit import Chem
from rdkit.Geometry import Point3D

from terrace.batch import Batch, Batchable, collate

class Pose(Batchable):
    coord: torch.Tensor

    @staticmethod
    def collate_coord(x):
        return x

def add_pose_to_mol(mol, pose):
    mol.RemoveAllConformers()
    conformer = Chem.Conformer(mol.GetNumAtoms())
    for i, coord in enumerate(pose.coord.detach().cpu()):
        conformer.SetAtomPosition(i, Point3D(float(coord[0]),
                                                float(coord[1]),
                                                float(coord[2])))

    mol.AddConformer(conformer)

# tried to make this class less hacky but only partially succeeded
# a single transform actually can emcompass a batch of transforms
# adding batches of batches to terrace will be complex tho, so rn
# we shall stick with this sadness
class PoseTransform(Batchable):

    # todo: torsional angles!
    rot: torch.Tensor
    trans: torch.Tensor

    @staticmethod
    def make_diffused(diff_cfg, timesteps, batch_size, device):
        schedule = torch.linspace(0.0, 1.0, timesteps, device=device).view((1,-1,1))**(diff_cfg.get("exponent", 1.0))
        trans_sigma = schedule*diff_cfg.max_trans_sigma
        trans = torch.randn((batch_size,timesteps,3), device=device)*trans_sigma
        
        rot_sigma = schedule*diff_cfg.max_rot_sigma
        rand_rot = torch.stack([torch.stack([roma.random_rotvec(device=device) for t in range(timesteps)]) for b in range(batch_size)])
        rot = rand_rot*rot_sigma
        return Batch(PoseTransform, rot=rot, trans=trans, trans_sigma=trans, rot_sigma=rot_sigma)

    @staticmethod
    def make_initial(diff_cfg, batch_size, device):
        trans_sigma = diff_cfg.max_trans_sigma
        trans = torch.randn((batch_size,1,3), device=device)*trans_sigma

        rot_sigma = diff_cfg.max_rot_sigma        
        rand_rot = torch.stack([torch.stack([roma.random_rotvec(device=device)]) for b in range(batch_size)])
        rot = rand_rot*rot_sigma
        return Batch(PoseTransform, rot=rot, trans=trans, trans_sigma=trans, rot_sigma=rot_sigma)

    def apply(self, pose):
        centroid = pose.coord.mean(0)
        rot_mat = roma.rotvec_to_rotmat(self.rot)
        return Pose(coord=torch.einsum('nij,bj->nbi', rot_mat, pose.coord - centroid) + self.trans.unsqueeze(1) + centroid)

    def batch_apply(self, lig_poses):
        return collate([ PoseTransform.apply(t,c) for t, c in zip(self, lig_poses) ])

    def grad(self, U):
        rot_grad, trans_grad = torch.autograd.grad(U, [self.rot, self.trans], create_graph=True)
        return PoseTransform(rot=rot_grad, trans=trans_grad)

    def batch_grad(self, U):
        rot_grad, trans_grad = torch.autograd.grad(U, [self.rot, self.trans], create_graph=True)
        return Batch(PoseTransform, rot=rot_grad, trans=trans_grad)

    def batch_requires_grad(self):
        self.rot.requires_grad_()
        self.trans.requires_grad_()

    def requires_grad(self):
        self.rot.requires_grad_()
        self.trans.requires_grad_()

    def batch_update_from_grad(self, grad):
        mul = 1.0
        trans_sigma_sq = (self.trans_sigma**2)
        rot_sigma_sq = (self.rot_sigma**2)
        rot = self.rot - mul*grad.rot # *rot_sigma_sq
        trans = self.trans - mul*grad.trans # *trans_sigma_sq
        # print(grad.trans.mean(), grad.rot.mean())
        return Batch(PoseTransform, rot=rot, trans=trans)