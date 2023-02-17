import torch
import torch.nn as nn
import torch.nn.functional as F
from common.pose_transform import PoseTransform, Pose
from models.force_field import ForceField
from terrace.batch import Batch, collate
from terrace.dataframe import DFRow, merge
from .model import Model
from validation.metrics import get_rmsds

from scipy.optimize import minimize

def get_transform_rmsds(x, y, transform):
    trans_poses = transform.apply(y.lig_crystal_pose, x.lig_torsion_data)
    ret = []
    for lig, tps, true_pose in zip(x.lig, trans_poses, y.lig_crystal_pose):
        rmsds = []
        for coord in tps.coord:
            rmsds.append(get_rmsds([lig], collate([Pose(coord)]), [true_pose])[0])
        ret.append(torch.stack(rmsds))
    return torch.stack(ret)

class Diffusion(nn.Module, Model):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.model.diffusion
        self.force_field = ForceField(cfg)

    @staticmethod
    def get_name() -> str:
        return "diffusion"

    def get_input_feats(self):
        return ["lig_embed_pose", "lig_torsion_data"] + self.force_field.get_input_feats()

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
        lig_poses = batch_transform.apply(batch_lig_pose, batch.lig_torsion_data)
        return self.force_field.get_energy(batch,
                                           batch_rec_feat,
                                           batch_lig_feat,
                                           lig_poses)
    def energy_jac_raw(self,
                       t_raw,
                       t_template,
                       batch,
                       batch_rec_feat,
                       batch_lig_feat,
                       batch_lig_pose):
        with torch.set_grad_enabled(True):
            device = batch_lig_feat.device
            t_raw = torch.tensor(t_raw, dtype=torch.float32, device=device)
            t_raw.requires_grad_()
            transform = PoseTransform.from_raw(t_raw, t_template)
            U = self.get_energy(batch,
                                batch_rec_feat,
                                batch_lig_feat,
                                batch_lig_pose,
                                transform)
            U_sum = U.sum()
            grad = torch.autograd.grad(U_sum, t_raw, create_graph=True)[0]
        return U_sum.cpu().numpy(), grad.cpu().numpy()       

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

            return transform.grad(U_sum) 

    def get_diffused_transforms(self, batch, device, timesteps=None):
        diff_cfg = self.cfg
        if timesteps is None:
            timesteps = diff_cfg.timesteps

        return PoseTransform.make_diffused(diff_cfg, timesteps, batch, device)

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

        transform = self.get_diffused_transforms(batch, device)

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

        transform = self.get_diffused_transforms(batch, device)

        return self.get_energy(batch, 
                              batch_rec_feat,
                              batch_lig_feat,
                              y.lig_crystal_pose,
                              transform), get_transform_rmsds(batch, y, transform)

    def infer_sgd(self, batch, hid_feat=None):
        """ Final inference -- predict lig_coords directly after randomizing """

        if hid_feat is None:
            batch_rec_feat, batch_lig_feat = self.get_hidden_feat(batch)
        else:
            batch_rec_feat, batch_lig_feat = hid_feat
        device = batch_lig_feat.device

        transform = PoseTransform.make_initial(self.cfg, batch, device)
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
            all_poses.append(Batch(Pose, coord=[p.coord[0] for p in transform.apply(init_pose, batch.lig_torsion_data)]))

        return [ collate([all_poses[i][j] for i in range(len(all_poses))]) for j in range(len(all_poses[0]))]

    def forward(self, batch, hid_feat=None):
        optim = self.cfg.get("optim", "bfgs")
        if optim == "sgd":
            lig_pose = collate([poses[-1] for poses in self.infer_sgd(batch, hid_feat)])
        elif optim == "bfgs":
            lig_pose, energy = self.infer_bfgs(batch, hid_feat)
            # return Batch(DFRow, lig_pose=lig_pose, energy=energy)
        return Batch(DFRow, lig_pose=lig_pose)

    def predict_train(self, x, y, task_names, batch_idx):
        hid_feat = self.get_hidden_feat(x)
        if self.cfg.get("energy_rmsd", False):
            diff_energy, diff_rmsds = self.diffuse_energy(x, y, hid_feat)
            ret_dif = Batch(DFRow, diffused_energy=diff_energy, diffused_rmsds=diff_rmsds)   
        else:
            diff = self.diffuse(x, y, hid_feat)
            diff_pose = diff.apply(y.lig_crystal_pose, x.lig_torsion_data)
            ret_dif = Batch(DFRow, diffused_transforms=diff, diffused_poses=diff_pose)
        if batch_idx % 50 == 0:
            with torch.no_grad():
                ret_pred = self(x, hid_feat)
            return merge([ret_dif, ret_pred])
        else:
            return ret_dif

    @torch.no_grad()
    def infer_bfgs(self, x, hid_feat = None, train=False):
        if hid_feat is None:
            batch_rec_feat, batch_lig_feat = self.get_hidden_feat(x)
        else:
            batch_rec_feat, batch_lig_feat = hid_feat

        device = batch_lig_feat.device
        best_energy = 1e10
        best_pose = None
        n_tries = 1 if train else self.cfg.get("optim_tries", 3)
        method = "BFGS"
        options = {
            # "disp": True,
            "maxiter": 20 if train else 30,
        }
        for i in range(n_tries):
            t = PoseTransform.make_initial(self.cfg, x, device)
            raw = PoseTransform.to_raw(t).cpu().numpy()
            init_pose = x.lig_embed_pose
            # print(model.energy_jac_raw(np.array(raw), x, batch_rec_feat, batch_lig_feat, init_pose))
            res = minimize(self.energy_jac_raw, raw, (t, x, batch_rec_feat, batch_lig_feat, init_pose), method=method, jac=True, options=options)
            opt_raw = torch.tensor(res.x, dtype=torch.float32, device=device)
            t_opt = PoseTransform.from_raw(opt_raw, t)
            pose = Batch(Pose, coord=[p.coord[0] for p in t_opt.apply(init_pose, x.lig_torsion_data)])
            if res.fun < best_energy:
                best_energy = res.fun
                best_pose = pose

        return best_pose, torch.tensor(best_energy, dtype=torch.float32, device=device)
