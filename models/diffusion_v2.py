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
from models.twister_v2 import TwistFFCoef, TwistForceField
from terrace.batch import Batch, collate
from terrace.dataframe import DFRow, merge
from .model import Model
from .diffusion import get_transform_rmsds
from validation.metrics import get_rmsds
from multiprocessing import Pool
import multiprocessing as mp

from scipy.optimize import minimize, basinhopping

class DiffusionV2(nn.Module, Model):
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.force_field = TwistForceField(cfg.model)

    @staticmethod
    def get_name() -> str:
        return "diffusion_v2"

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
                   hid_feat,
                   batch_lig_pose,
                   batch_transform):
        lig_poses = batch_transform.apply(batch_lig_pose, batch.lig_torsion_data)
        return self.force_field.get_energy(batch,
                                           hid_feat,
                                           lig_poses)

    @staticmethod
    def energy_raw(t_raw,
                   cfg,
                   rot_edges,
                   rot_masks,
                   ll_coef,
                   l_rf_coef,
                   init_lig_coord,
                   rec_coord,
                   weight,
                   bias):
        # tor_data = tor_data_wrap.obj
        transform = PoseTransform.from_raw(t_raw)
        tor_data = TorsionData(rot_edges, rot_masks)
        lig_pose = transform.apply(Pose(coord=init_lig_coord), tor_data)
        U = TwistForceField.get_energy_single(cfg.model,
                                        ll_coef,
                                        l_rf_coef,
                                        rec_coord,
                                        lig_pose.coord,
                                        weight,
                                        bias)
        return U

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

    def diffuse_energy(self, batch, y, hid_feat=None):
        
        if hid_feat is None:
            hid_feat= self.get_hidden_feat(batch)
        device = batch.lig_graph.ndata.cat_feat.device

        transform = self.get_diffused_transforms(batch, device)

        if hasattr(y, "lig_crystal_pose"):
            true_pose = y.lig_crystal_pose
        else:
            true_pose = collate([Pose(p.coord[0]) for p in batch.lig_docked_poses])

        energy = self.get_energy(batch,
                                 hid_feat,
                                 true_pose,
                                 transform)

        if self.cfg.model.diffusion.pred == "rank":
            rmsds = torch.linspace(0,1,energy.shape[1], device=energy.device).unsqueeze(0).repeat(energy.shape[0], 1)
        else:
            assert self.cfg.model.diffusion.pred == "rmsd"
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
        diff_energy, diff_rmsds = self.diffuse_energy(x, y, hid_feat)
        ret_dif = Batch(DFRow, diffused_energy=diff_energy, diffused_rmsds=diff_rmsds)
        if "predict_lig_pose" in task_names and (split != "train" or batch_idx % self.cfg.metric_reset_interval == 0):
            with torch.no_grad():
                ret_pred = self(x, hid_feat)
                # pred_crystal = self(x, hid_feat, self.get_true_pose(y))
                # pred_crystal = Batch(DFRow, crystal_pose=pred_crystal.lig_pose.get(0), crystal_energy=pred_crystal.energy[:,0])
                # if self.cfg.model.diffusion.get("only_pred_local_min", False):
                #     ret_pred = pred_crystal
                # else:
                #     ret_pred = self(x, hid_feat)
                #     ret_pred = merge([ret_pred, pred_crystal])
            return merge([ret_dif, ret_pred])
        else:
            return ret_dif

    @staticmethod
    def get_init_pose(x):
        init_poses = []
        for x0 in x:
            embed_pose = x0.lig_docked_poses.get(0)
            rec_mean = x0.full_rec_data.ndata.coord.mean(0)
            lig_mean = embed_pose.coord.mean(0)
            init_poses.append(Pose(coord=embed_pose.coord + rec_mean - lig_mean))
        return collate(init_poses)

    @torch.no_grad()
    def infer_bfgs(self, x, hid_feat = None, init_pose_override=None):
        # x = deepcopy(x)
        if hid_feat is None:
            hid_feat = self.get_hidden_feat(x)
        device = x.lig_graph.ndata.cat_feat.device

        bias, weight = self.force_field.scale_output.parameters()

        if init_pose_override is None:
            # print("running init energy")
            num_rand_poses = self.cfg.model.diffusion.get("num_init_poses", 64)
            init_pose = DiffusionV2.get_init_pose(x)
            rand_transforms = PoseTransform.make_initial(self.cfg.model.diffusion, x, device, num_rand_poses)
            rand_poses = rand_transforms.apply(init_pose, x.lig_torsion_data).cpu()
            init_energy = self.get_energy(x, hid_feat, init_pose, rand_transforms).cpu()
            # print("stopped init energy")
        else:
            rand_poses = [None]*len(x)
            init_energy = [None]*len(x)

        hid_feat = hid_feat.cpu()
        x = x.cpu()
        ret = []
        args = []
        for i in range(len(x)):

            if init_pose_override is not None:
                pose_override = init_pose_override.cpu()[i]
            else:
                pose_override = None

            args.append((self.cfg, x[i], hid_feat[i], weight.detach().cpu(), bias.detach().cpu(), pose_override, rand_poses[i], init_energy[i]))
        
        if not hasattr(DiffusionV2, "jit_infer"):
            if self.cfg.model.diffusion.get("use_jit", True):
                DiffusionV2.jit_infer = jax.jit(jax.value_and_grad(to_jax(DiffusionV2.energy_raw)), static_argnums=1)
            else:
                DiffusionV2.jit_infer = jax.value_and_grad(to_jax(DiffusionV2.energy_raw))
            # DiffusionV2.infer_bfgs_single((*args[0][:-1], True))

        if self.cfg.platform.infer_workers > 0:
            # with ThreadPoolExecutor(max_workers=self.cfg.platform.infer_workers) as p:
            with Pool(self.cfg.platform.infer_workers) as p:
                for res in p.imap(DiffusionV2.infer_bfgs_single, args):
                    ret.append(res)
        else:
            for arg in args:
                ret.append(DiffusionV2.infer_bfgs_single(arg))

        return collate(ret)

    @staticmethod
    def infer_bfgs_single(args):
        cfg, x, hid_feat, weight, bias, init_pose_override, rand_poses, init_energy = args
        # f = jax.jit(jax.value_and_grad(to_jax(DiffusionV2.energy_raw)), static_argnums=1) # deepcopy(DiffusionV2.jit_infer)
        f = DiffusionV2.jit_infer

        method = "BFGS"
        options = {
            # "disp": True,
            "maxiter": 30,
        }

        pose_and_energies = []
        n_tries = cfg.model.diffusion.get("optim_tries", 16)
        if init_pose_override is not None:
            n_tries = 1
        if cfg.model.get("fix_infer_torsion", False):
            n_tries = cfg.data.num_poses

        if init_pose_override is None:
            best_pose_indices = init_energy.argsort()
            assert len(best_pose_indices) >= n_tries

        for i in range(n_tries):  
            if init_pose_override is None:
                pose_index = best_pose_indices[i]
                init_pose = rand_poses.get(pose_index)
                # pose_idx = min(x.lig_docked_poses.coord.shape[0]-1, i)
                # embed_pose = x.lig_docked_poses.get(pose_idx)
                # rec_mean = x.full_rec_data.coord.mean(0)
                # lig_mean = embed_pose.coord.mean(0)
                # init_pose = Pose(coord=embed_pose.coord + rec_mean - lig_mean)
            else:
                init_pose = init_pose_override

            if cfg.model.get("fix_infer_torsion", False):
                x.lig_torsion_data.rot_edges = torch.zeros((0,0))
                x.lig_torsion_data.rot_masks = torch.zeros((0,0))

            extra_args = (
                cfg,
                x.lig_torsion_data.rot_edges.detach().cpu().numpy(),
                x.lig_torsion_data.rot_masks.detach().cpu().numpy(),
                hid_feat.ll_coef.numpy(),
                hid_feat.l_rf_coef.numpy(),
                init_pose.coord.detach().cpu().numpy(),
                x.full_rec_data.ndata.coord.detach().cpu().numpy(),
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
            opt_raw = torch.tensor(res.x, dtype=torch.float32, device='cpu')
            # t_opt = PoseTransform.from_raw(torch.tensor(raw, dtype=torch.float32, device=device))
            t_opt = PoseTransform.from_raw(opt_raw)
            pose = t_opt.apply(init_pose, x.lig_torsion_data)
            pose_and_energies.append((res.fun, pose))

        poses = []
        energies = []
        for energy, pose in sorted(pose_and_energies, key=lambda x: x[0]):
            energies.append(energy)
            poses.append(pose)

        return MultiPose.combine(poses), torch.asarray(energies)