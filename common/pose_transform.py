import roma
import torch
from rdkit import Chem
from rdkit.Geometry import Point3D
from functools import reduce

from terrace.batch import Batch, Batchable, collate

class Pose(Batchable):
    coord: torch.Tensor

    @staticmethod
    def collate_coord(x):
        return x

class MultiPose(Batchable):
    coord: torch.Tensor

    @staticmethod
    def collate_coord(x):
        return x

    @staticmethod
    def combine(poses):
        return MultiPose(torch.stack([ p.coord for p in poses ]))

    def get(self, i):
        return Pose(coord=self.coord[i])

    def items(self):
        for coord in self.coord:
            yield Pose(coord)

    def batch_get(self, i):
        return collate([mp.get(i) for mp in self.items()])

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
    tor_angles: torch.Tensor

    @staticmethod
    def collate_tor_angles(x):
        return x

    @staticmethod
    def to_raw(t):
        buf = []
        buf.append(t.trans.reshape(-1))
        buf.append(t.rot.reshape(-1))
        for angle in t.tor_angles:
            buf.append(angle.reshape(-1))
        return torch.cat(buf, 0)

    @staticmethod
    def from_raw(raw, template=None):
        idx = 0
        def eat_arr(temp_arr):
            nonlocal idx
            shape = temp_arr.shape
            size = reduce(lambda a, b: a*b, shape)
            ret = raw[idx:idx+size].reshape(shape)
            idx += size
            return ret
        if template is None:
            trans = raw[:3]
            rot = raw[3:6]
            angles = raw[6:]
        else:
            trans = eat_arr(template.trans)
            rot = eat_arr(template.rot)
            angles = []
            for angle in template.tor_angles:
                angles.append(eat_arr(angle))
            if not isinstance(template, Batch):
                angles = torch.stack(angles)
        if isinstance(template, Batch):
            return Batch(PoseTransform, trans=trans, rot=rot, tor_angles=angles)
        else:
            return PoseTransform(trans=trans, rot=rot, tor_angles=angles)

    @staticmethod
    def make_diffused(diff_cfg, timesteps, batch, device):
        batch_size = len(batch)
        schedule = torch.linspace(0.0, 1.0, timesteps, device=device).view((1,-1,1))**(diff_cfg.get("exponent", 1.0))
        trans_sigma = schedule*diff_cfg.max_trans_sigma
        trans = torch.randn((batch_size,timesteps,3), device=device)*trans_sigma

        rand_rot = torch.stack([torch.stack([roma.random_rotvec(device=device) for t in range(timesteps)]) for b in range(batch_size)])
        if hasattr(diff_cfg, "max_rot_sigma"):
            rot_sigma = schedule*diff_cfg.max_rot_sigma
            rot = rand_rot*torch.randn((batch_size, timesteps, 1), device=device)*rot_sigma
        else:
            rot = rand_rot*schedule

        tor_sigma = schedule.view(-1,1)*diff_cfg.max_tor_sigma
        angles = []
        for tor_data in batch.lig_torsion_data:
            angles.append(torch.randn((timesteps, len(tor_data.rot_edges)), device=device)*tor_sigma)

        return Batch(PoseTransform, rot=rot, trans=trans, tor_angles=angles)

    @staticmethod
    def make_initial(diff_cfg, batch, device):
        batch_size = len(batch)
        trans_sigma = diff_cfg.max_trans_sigma
        trans = torch.randn((batch_size,1,3), device=device)*trans_sigma
    
        rot = torch.stack([torch.stack([roma.random_rotvec(device=device)]) for b in range(batch_size)])
        if hasattr(diff_cfg, "max_rot_sigma") and diff_cfg.max_rot_sigma == 0:
            rot = torch.zeros_like(rot)

        angles = []
        for tor_data in batch.lig_torsion_data:
            angles.append(torch.rand((1, len(tor_data.rot_edges)), device=device)*2*torch.pi)

        return Batch(PoseTransform, rot=rot, trans=trans, tor_angles=angles)

    def apply(self, pose, tor_data, use_tor=True):
        coord = pose.coord
        rot_mat = roma.rotvec_to_rotmat(self.rot)
        if self.tor_angles.size(-1) > 0 and use_tor:
            coord = tor_data.set_all_angles(self.tor_angles, coord)
        centroid = coord.mean(-2, keepdim=True)
        
        if len(coord.shape) == 3 or len(rot_mat.shape) == 3:
            trans = self.trans.unsqueeze(1)
        else:
            trans = self.trans
        coord = torch.einsum('...ij,...bj->...bi', rot_mat, coord - centroid) + trans + centroid
        return Pose(coord=coord)

    def batch_apply(self, lig_poses, lig_tor_data):
        return collate([ PoseTransform.apply(t,c,d) for t,c,d in zip(self, lig_poses, lig_tor_data) ])

    def grad(self, U):
        rot_grad, trans_grad = torch.autograd.grad(U, [self.rot, self.trans], create_graph=True)
        return PoseTransform(rot=rot_grad, trans=trans_grad)

    def batch_grad(self, U):
        # todo: why allow unused?
        rot_grad, trans_grad, *tor_grad = torch.autograd.grad(U, [self.rot, self.trans, *self.tor_angles], create_graph=True, allow_unused=True)
        return Batch(PoseTransform, rot=rot_grad, trans=trans_grad, tor_angles=tor_grad)

    def batch_requires_grad(self):
        self.rot.requires_grad_()
        self.trans.requires_grad_()
        for angle in self.tor_angles:
            angle.requires_grad_()

    def requires_grad(self):
        self.rot.requires_grad_()
        self.trans.requires_grad_()
        self.tor_angles.requires_grad_()

    def batch_update_from_grad(self, grad):
        mul = 1.0
        rot = self.rot - mul*grad.rot
        trans = self.trans - mul*grad.trans
        tor = []
        for angle, grad_angle in zip(self.tor_angles, grad.tor_angles):
            if grad_angle is None:
                tor.append(angle)
            else:
                tor.append(angle - mul*grad_angle)
        # print(grad.rot, grad.tor_angles)
        return Batch(PoseTransform, rot=rot, trans=trans, tor_angles=tor)