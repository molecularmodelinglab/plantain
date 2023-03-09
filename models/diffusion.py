from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import jax
from common.pose_transform import MultiPose, PoseTransform, Pose
from common.torsion import TorsionData
from common.jorch import to_jax
from models.force_field import ForceField
from terrace.batch import Batch, collate
from terrace.dataframe import DFRow, merge
from .model import Model
from validation.metrics import get_rmsds
from multiprocessing import Pool
import multiprocessing as mp

from scipy.optimize import minimize, basinhopping

jax.config.update('jax_platform_name', 'cpu')

class Wrapper:
    def __init__(self, obj):
        self.obj = obj

def get_transform_rmsds(x, true_pose, transform):
    trans_poses = transform.apply(true_pose, x.lig_torsion_data)
    ret = []
    for lig, tps, true_pose in zip(x.lig, trans_poses, true_pose):
        rmsds = []
        for coord in tps.coord:
            rmsds.append(get_rmsds([lig], collate([Pose(coord)]), [true_pose])[0])
        ret.append(torch.stack(rmsds))
    return torch.stack(ret)

class Diffusion(nn.Module, Model):
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.force_field = ForceField(cfg)

    @staticmethod
    def get_name() -> str:
        return "diffusion"

    def get_input_feats(self):

        ret = ["lig_embed_pose", "lig_torsion_data"] + self.force_field.get_input_feats()
        if "pose_sample" in self.cfg.data:
            ret.append("lig_docked_poses")
        return ret

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

    @staticmethod
    def energy_raw(t_raw,
                   cfg,
                   # tor_data_wrap,
                   rot_edges,
                   rot_masks,
                   lig_feat,
                   rec_feat,
                   init_lig_coord,
                   rec_coord,
                   weight,
                   bias):
        # tor_data = tor_data_wrap.obj
        transform = PoseTransform.from_raw(t_raw)
        tor_data = TorsionData(rot_edges, rot_masks)
        lig_pose = transform.apply(Pose(coord=init_lig_coord), tor_data)
        U = ForceField.get_energy_single(cfg.model,
                                        rec_feat,
                                        lig_feat,
                                        rec_coord,
                                        lig_pose.coord,
                                        weight,
                                        bias)
        return U

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
        diff_cfg = self.cfg.model.diffusion
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

    def get_true_pose(self, y):
        if self.cfg.data.get("use_embed_crystal_pose", False):
            return y.lig_embed_crystal_pose
        else:
            return y.lig_crystal_pose

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
                              self.get_true_pose(y),
                              transform)

    def diffuse_energy(self, batch, y, hid_feat=None):
        
        if hid_feat is None:
            batch_rec_feat, batch_lig_feat = self.get_hidden_feat(batch)
        else:
            batch_rec_feat, batch_lig_feat = hid_feat
        device = batch_lig_feat.device

        transform = self.get_diffused_transforms(batch, device)

        if hasattr(y, "lig_crystal_pose"):
            true_pose = y.lig_crystal_pose
        else:
            true_pose = collate([Pose(p.coord[0]) for p in batch.lig_docked_poses])

        energy = self.get_energy(batch, 
                              batch_rec_feat,
                              batch_lig_feat,
                              true_pose,
                              transform)

        # if self.cfg.model.diffusion.get("energy_rank", False):
        #     rmsds = torch.linspace(0,1,energy.shape[1]).unsqueeze(0).repeat(energy.shape[0], 1)
        # else:
        rmsds = get_transform_rmsds(batch, true_pose, transform)

        return energy, rmsds

    def forward(self, batch, hid_feat=None, init_pose_override=None):
        optim = "bfgs"
        if optim == "bfgs":
            lig_pose, energy = self.infer_bfgs(batch, hid_feat, init_pose_override)
            # return Batch(DFRow, lig_pose=lig_pose, energy=energy)
        return Batch(DFRow, lig_pose=lig_pose, energy=energy)

    def predict_train(self, x, y, task_names, split, batch_idx):
        hid_feat = self.get_hidden_feat(x)
        if self.cfg.model.diffusion.get("energy_rmsd", False):
            diff_energy, diff_rmsds = self.diffuse_energy(x, y, hid_feat)
            ret_dif = Batch(DFRow, diffused_energy=diff_energy, diffused_rmsds=diff_rmsds)   
        else:
            diff = self.diffuse(x, y, hid_feat)
            diff_pose = diff.apply(self.get_true_pose(y), x.lig_torsion_data)
            ret_dif = Batch(DFRow, diffused_transforms=diff, diffused_poses=diff_pose)
        if "predict_lig_pose" in task_names and (split != "train" or batch_idx % 50 == 0):
            with torch.no_grad():
                ret_pred = self(x, hid_feat)
                # pred_crystal = self(x, hid_feat, self.get_true_pose(y))
                # pred_crystal = Batch(DFRow, crystal_pose=pred_crystal.lig_pose.get(0), crystal_energy=pred_crystal.energy[:,0])
                # ret_pred = merge([ret_pred, pred_crystal])
            return merge([ret_dif, ret_pred])
        else:
            return ret_dif

    @torch.no_grad()
    def infer_bfgs(self, x, hid_feat = None, init_pose_override=None):
        if hid_feat is None:
            batch_rec_feat, batch_lig_feat = self.get_hidden_feat(x)
        else:
            batch_rec_feat, batch_lig_feat = hid_feat

        bias, weight = self.force_field.scale_output.parameters()

        x = x.cpu()
        ret = []
        args = []
        tot_lig = 0
        for i, l in enumerate(x.lig_graph.dgl().batch_num_nodes()):

            lig_feat = batch_lig_feat[tot_lig:tot_lig+l]
            rec_feat = batch_rec_feat[i]
            tot_lig += l

            if init_pose_override is not None:
                pose_override = init_pose_override.cpu()[i]
            else:
                pose_override = None

            args.append((self.cfg, x[i], rec_feat.detach().cpu(), lig_feat.detach().cpu(), weight.detach().cpu(), bias.detach().cpu(), pose_override))
        
        if not hasattr(Diffusion, "jit_infer"):
            Diffusion.jit_infer = jax.jit(jax.value_and_grad(to_jax(Diffusion.energy_raw)), static_argnums=1)
            # Diffusion.infer_bfgs_single((*args[0][:-1], True))

        if self.cfg.platform.infer_workers > 0:
            # with ThreadPoolExecutor(max_workers=self.cfg.platform.infer_workers) as p:
            with Pool(self.cfg.platform.infer_workers) as p:
                for res in p.imap(Diffusion.infer_bfgs_single, args):
                    ret.append(res)
        else:
            for arg in args:
                ret.append(Diffusion.infer_bfgs_single(arg))

        return collate(ret)

    @staticmethod
    def infer_bfgs_single(args):
        cfg, x, rec_feat, lig_feat, weight, bias, init_pose_override = args
        # f = jax.jit(jax.value_and_grad(to_jax(Diffusion.energy_raw)), static_argnums=1) # deepcopy(Diffusion.jit_infer)
        f = Diffusion.jit_infer

        method = "BFGS"
        options = {
            # "disp": True,
            "maxiter": 30,
        }
        
        device = lig_feat.device

        pose_and_energies = []
        n_tries = cfg.model.diffusion.get("optim_tries", 16)
        if init_pose_override is not None:
            n_tries = 1
        if cfg.model.get("fix_infer_torsion", False):
            n_tries = cfg.data.num_poses

        for i in range(n_tries):  
            
            if init_pose_override is None:
                pose_idx = min(x.lig_docked_poses.coord.shape[0]-1, i)
                embed_pose = x.lig_docked_poses.get(pose_idx)
                rec_mean = x.full_rec_data.coord.mean(0)
                lig_mean = embed_pose.coord.mean(0)
                init_pose = Pose(coord=embed_pose.coord + rec_mean - lig_mean)
            else:
                init_pose = init_pose_override

            if cfg.model.get("fix_infer_torsion", False):
                x.lig_torsion_data.rot_edges = torch.zeros((0,0))
                x.lig_torsion_data.rot_masks = torch.zeros((0,0))

            extra_args = (
                cfg,
                x.lig_torsion_data.rot_edges.detach().cpu().numpy(),
                x.lig_torsion_data.rot_masks.detach().cpu().numpy(),
                lig_feat.detach().cpu().numpy(),
                rec_feat.detach().cpu().numpy(),
                init_pose.coord.detach().cpu().numpy(),
                x.full_rec_data.coord.detach().cpu().numpy(),
                weight.detach().cpu().numpy(),
                bias.detach().cpu().numpy(),
            )


            t = PoseTransform.make_initial(cfg.model.diffusion, collate([x]), 'cpu')[0]
            raw = PoseTransform.to_raw(t)
            if init_pose_override is not None:
                raw = torch.zeros_like(raw) + torch.randn_like(raw)*0.1
            raw = raw.numpy()

            res = minimize(f, raw, extra_args, method=method, jac=True, options=options)
            # res = basinhopping(f, raw, niter=32, T=3.0, stepsize=3.0, minimizer_kwargs={"method": method, "jac": True, "args": extra_args, "options": options})
            opt_raw = torch.tensor(res.x, dtype=torch.float32, device=device)
            # t_opt = PoseTransform.from_raw(torch.tensor(raw, dtype=torch.float32, device=device))
            t_opt = PoseTransform.from_raw(opt_raw)
            # print(opt_raw)
            pose = t_opt.apply(init_pose, x.lig_torsion_data)
            pose_and_energies.append((res.fun, pose))

        poses = []
        energies = []
        for energy, pose in sorted(pose_and_energies, key=lambda x: x[0]):
            energies.append(energy)
            poses.append(pose)

        return MultiPose.combine(poses), torch.asarray(energies)