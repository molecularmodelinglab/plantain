import torch
import torch.nn as nn
import torch.nn.functional as F
from common.pose_transform import PoseTransform, Pose
from models.force_field import ForceField
from terrace.batch import Batch, collate
from terrace.dataframe import DFRow, merge
from .model import Model
from validation.metrics import get_rmsds

# def get_transform_rmsds(x, y, transform):
#     trans_poses = transform.apply(y.lig_crystal_pose)
#     ret = []
#     for lig, tps, true_pose in zip(x.lig, trans_poses, y.lig_crystal_pose):


class Diffusion(nn.Module, Model):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.model.diffusion
        self.force_field = ForceField(cfg)

    @staticmethod
    def get_name() -> str:
        return "diffusion"

    def get_input_feats(self):
        return ["lig_embed_pose"] + self.force_field.get_input_feats()

    def get_tasks(self):
        return ["predict_lig_pose"]
        
    def get_hidden_feat(self, x):
        return self.force_field.get_hidden_feat(x)

    def get_energy(self,
                   batch,
                   batch_rec_feat,
                   batch_lig_feat,
                   batch_lig_pose,
                   batch_transform):
        lig_poses = batch_transform.apply(batch_lig_pose)
        return self.force_field.get_energy(batch,
                                           batch_rec_feat,
                                           batch_lig_feat,
                                           lig_poses)

    def energy_grad(self,
                    batch,
                    batch_rec_feat,
                    batch_lig_feat,
                    batch_lig_pose,
                    transform):

        with torch.set_grad_enabled(True):
            transform.requires_grad()
            U = self.get_energy(batch,
                                batch_rec_feat,
                                batch_lig_feat,
                                batch_lig_pose,
                                transform)
            U_sum = U.sum()
            # print(U_sum)

            return transform.grad(U_sum) 

    def get_diffused_transforms(self, batch_size, device, timesteps=None):
        diff_cfg = self.cfg
        if timesteps is None:
            timesteps = diff_cfg.timesteps

        return PoseTransform.make_diffused(diff_cfg, timesteps, batch_size, device)

    def pred_pose(self,
                  batch,
                  batch_rec_feat,
                  batch_lig_feat,
                  batch_lig_pose,
                  transform):

        grad = self.energy_grad(batch,
                                batch_rec_feat,
                                batch_lig_feat,
                                batch_lig_pose,
                                transform)
        return transform.update_from_grad(grad)

    def diffuse(self, batch, y, hid_feat=None):

        if hid_feat is None:
            batch_rec_feat, batch_lig_feat = self.get_hidden_feat(batch)
        else:
            batch_rec_feat, batch_lig_feat = hid_feat
        device = batch_lig_feat.device

        transform = self.get_diffused_transforms(len(batch), device)

        return self.pred_pose(batch, 
                              batch_rec_feat,
                              batch_lig_feat,
                              y.lig_crystal_pose,
                              transform)

    def diffuse_energy(self, batch, y, hid_feat=None):
        
        if hid_feat is None:
            batch_rec_feat, batch_lig_feat = self.get_hidden_feat(batch)
        else:
            batch_rec_feat, batch_lig_feat = hid_feat
        device = batch_lig_feat.device

        transform = self.get_diffused_transforms(len(batch), device)

        return self.get_energy(batch, 
                              batch_rec_feat,
                              batch_lig_feat,
                              y.lig_crystal_pose,
                              transform)

    def infer(self, batch, hid_feat=None):
        """ Final inference -- predict lig_coords directly after randomizing """

        if hid_feat is None:
            batch_rec_feat, batch_lig_feat = self.get_hidden_feat(batch)
        else:
            batch_rec_feat, batch_lig_feat = hid_feat
        device = batch_lig_feat.device

        transform = PoseTransform.make_initial(self.cfg, len(batch), device)
        init_pose = batch.lig_embed_pose

        trans_sigma = torch.linspace(0.0, self.cfg.max_trans_sigma, self.cfg.timesteps, device=device).view((1,-1,1))
        rot_sigma = torch.linspace(0.0, self.cfg.max_rot_sigma, self.cfg.timesteps, device=device).view((1,-1,1))

        all_poses = []
        for t in range(self.cfg.timesteps):
            transform.trans_sigma = trans_sigma[:,-1-t]
            transform.rot_sigma = rot_sigma[:,-1-t]
            transform = self.pred_pose(batch, 
                                       batch_rec_feat,
                                       batch_lig_feat,
                                       init_pose,
                                       transform)
            all_poses.append(Batch(Pose, coord=[p.coord[0] for p in transform.apply(init_pose)]))

        return [ collate([all_poses[i][j] for i in range(len(all_poses))]) for j in range(len(all_poses[0]))]

    def forward(self, batch, hid_feat=None):
        lig_pose = collate([poses[-1] for poses in self.infer(batch, hid_feat)])
        return Batch(DFRow, lig_pose=lig_pose)

    def predict_train(self, x, y, task_names, batch_idx):
        hid_feat = self.get_hidden_feat(x)
        diff = self.diffuse(x, y, hid_feat)
        diff_pose = diff.apply(y.lig_crystal_pose)
        ret_dif = Batch(DFRow, diffused_transforms=diff, diffused_poses=diff_pose)
        if batch_idx % 50 == 0:
            with torch.no_grad():
                ret_pred = self(x, hid_feat)
            return merge([ret_dif, ret_pred])
        else:
            return ret_dif