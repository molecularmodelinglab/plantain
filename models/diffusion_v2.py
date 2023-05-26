from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import dgl.backend as dF
import jax
from common.pose_transform import MultiPose, PoseTransform, Pose
from common.torsion import TorsionData
from common.jorch import to_jax
from models.twister_v2 import TwistFFCoef, TwistForceField
from terrace.batch import Batch, Batchable, collate
from terrace.dataframe import DFRow, merge
from .model import Model
from .diffusion import get_transform_rmsds
from validation.metrics import get_rmsds
from multiprocessing import Pool
import multiprocessing as mp

from scipy.optimize import minimize, basinhopping

class DiffPred(Batchable):
    """ Need custom collates for padded tensors. Todo: make this a more general class 
    (possible put in Terrace)"""
    diffused_energy: torch.Tensor
    diffused_rmsds: torch.Tensor
    inv_dist_mat: torch.Tensor

    @staticmethod
    def collate_diffused_energy(tensor_list, dims=[0]):
        max_xs = [ max([t.shape[dim] for t in tensor_list]) for dim in dims ]
        padded_tensors = []
        for t in tensor_list:
            padded_tensor = t
            for max_x, dim in zip(max_xs, dims):
                pad_length = max_x - t.shape[dim]
                # yes, this is a cursed line
                # but it's not my fault F.pad has a horrible api
                padded_tensor = F.pad(padded_tensor, (*([0,0]*(t.dim()-dim-1)), 0, pad_length))
            padded_tensors.append(padded_tensor)
        stacked_tensors = torch.stack(padded_tensors)
        return stacked_tensors

    @staticmethod
    def collate_diffused_rmsds(tensor_list):
        return DiffPred.collate_diffused_energy(tensor_list)

    @staticmethod
    def collate_inv_dist_mat(tensor_list):
        return DiffPred.collate_diffused_energy(tensor_list, [0,1])

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
                   lig_poses,
                   per_atom_energy=False):
        energy = self.force_field.get_energy(batch,
                                            hid_feat,
                                            lig_poses,
                                            per_atom_energy)
        assert per_atom_energy
        # V1 energy returns packed, V2 energy returns padded
        if "energy" in self.cfg.model:
            energy = energy.transpose(1,2).contiguous()
        else:
            energy = dF.pad_packed_tensor(energy if energy.dim() == 1 else energy.transpose(0,1), batch.lig_graph.dgl().batch_num_nodes(), 0.0)

        return energy

    @staticmethod
    def energy_raw(t_raw,
                   cfg,
                   rot_edges,
                   rot_masks,
                   ll_coef,
                   l_rf_coef,
                   init_lig_coord,
                   rec_coord,
                   *params):
        # tor_data = tor_data_wrap.obj
        transform = PoseTransform.from_raw(t_raw)
        tor_data = TorsionData(rot_edges, rot_masks)
        lig_pose = transform.apply(Pose(coord=init_lig_coord), tor_data)
        U = TwistForceField.get_energy_single(cfg.model,
                                        ll_coef,
                                        l_rf_coef,
                                        rec_coord,
                                        lig_pose.coord,
                                        *params)
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

        if hasattr(y, "lig_crystal_pose"):
            true_pose = y.lig_crystal_pose
        else:
            true_pose = collate([Pose(p.coord[0]) for p in batch.lig_docked_poses])

        transform = self.get_diffused_transforms(batch, device)
        diff_pose = transform.apply(true_pose, batch.lig_torsion_data)

        per_atom_energy = self.cfg.model.diffusion.pred == "atom_dist"
        energy = self.get_energy(batch,
                                 hid_feat,
                                 diff_pose,
                                 per_atom_energy)

        if self.cfg.model.diffusion.pred == "atom_dist":
            diff_coord = torch.cat(diff_pose.coord, 1)
            true_coord= torch.cat(true_pose.coord, 0).unsqueeze(0)
            dists = torch.linalg.norm(diff_coord - true_coord, dim=-1)
            rmsds = dists
            rmsds = dF.pad_packed_tensor(rmsds.transpose(0,1), batch.lig_graph.dgl().batch_num_nodes(), 0.0)

        elif self.cfg.model.diffusion.pred == "rank":
            rmsds = torch.linspace(0,1,energy.shape[1], device=energy.device).unsqueeze(0).repeat(energy.shape[0], 1)
        else:
            assert self.cfg.model.diffusion.pred == "rmsd"
            rmsds = get_transform_rmsds(batch, true_pose, transform)

        return energy, rmsds, hid_feat.inv_dist_mat

    def forward(self, batch, hid_feat=None, init_pose_override=None):
        # todo: remove timer here. This is not a place of benchmarking
        # start = time.time()
        optim = "bfgs"
        if optim == "bfgs":
            lig_pose, energy = self.infer_bfgs(batch, hid_feat, init_pose_override)
            # return Batch(DFRow, lig_pose=lig_pose, energy=energy)
        # end = time.time()
        # if hasattr(batch, "index"):
        #     with open("outputs/plantain_timer.txt", "a") as f:
        #         f.write(f"{int(batch.index)},{end-start}\n")
        return Batch(DFRow, lig_pose=lig_pose, energy=energy)

    def predict_train(self, x, y, task_names, split, batch_idx):
        hid_feat = self.get_hidden_feat(x)
        diff_energy, diff_rmsds, inv_dist_mat = self.diffuse_energy(x, y, hid_feat)
        ret_dif = Batch(DiffPred, diffused_energy=diff_energy, diffused_rmsds=diff_rmsds, inv_dist_mat=inv_dist_mat)
        # ret_dif = Batch(DiffPred, diffused_energy=diff_energy, diffused_rmsds=diff_rmsds, inv_dist_mat=inv_dist_mat, hid_feat=hid_feat)
        if "predict_lig_pose" in task_names and (split != "train" or batch_idx % self.cfg.metric_reset_interval == 0):
            with torch.no_grad():
                # ret_pred = self(x, hid_feat)
                pred_crystal = self(x, hid_feat, self.get_true_pose(y))
                pred_crystal = Batch(DFRow, crystal_pose=pred_crystal.lig_pose.get(0), crystal_energy=pred_crystal.energy[:,0])
                if self.cfg.model.diffusion.get("only_pred_local_min", False):
                    ret_pred = pred_crystal
                else:
                    ret_pred = self(x, hid_feat)
                    ret_pred = merge([ret_pred, pred_crystal])
            # oo now this is cursed. CLearly we need to change some of terrace's API
            # to do what I want to do
            ret = merge([ret_dif, ret_pred])
            ret._batch_type = DiffPred
            return ret
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

        per_atom_energy = self.cfg.model.diffusion.pred == "atom_dist"

        if "energy" in self.cfg.model:
            params = self.force_field.get_energy_v2_params()
        else:
            bias, weight = self.force_field.scale_output.parameters()
            params = [ weight, bias ]
        params = [ p.detach().cpu() for p in params ]

        if init_pose_override is None:
            # print("running init energy")
            num_rand_poses = self.cfg.model.diffusion.get("num_init_poses", 64)
            init_pose = DiffusionV2.get_init_pose(x)
            rand_transforms = PoseTransform.make_initial(self.cfg.model.diffusion, x, device, num_rand_poses)
            rand_poses = rand_transforms.apply(init_pose, x.lig_torsion_data)
            init_energy = self.get_energy(x, hid_feat, rand_poses, per_atom_energy).cpu()
            if per_atom_energy:
                init_energy = init_energy.sum(1)
            rand_poses = rand_poses.cpu()
            # print("stopped init energy")
        else:
            rand_poses = [None]*len(x)
            init_energy = [None]*len(x)

        x = x.cpu()
        ret = []
        args = []
        for i, (L, Rf) in enumerate(zip(x.lig_graph.dgl().batch_num_nodes(),
                                        x.full_rec_data.dgl().batch_num_nodes())):

            if init_pose_override is not None:
                pose_override = init_pose_override.cpu()[i]
            else:
                pose_override = None

            # ensure we unpad hid_feat before sending to the bfgs
            # todo: this should be done automagically upon indexing...
            hid_feati = DFRow(
                        l_rf_coef=hid_feat.l_rf_coef.detach().cpu()[i,:L,:Rf],
                        ll_coef=hid_feat.ll_coef.detach().cpu()[i,:L,:L]
                        )

            args.append((self.cfg, x[i], hid_feati, params, pose_override, rand_poses[i], init_energy[i]))
        
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
        cfg, x, hid_feat, params, init_pose_override, rand_poses, init_energy = args
        # f = jax.value_and_grad(to_jax(DiffusionV2.energy_raw))
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
                *[ p.detach().cpu().numpy() for p in params ]
            )


            t = PoseTransform.make_initial(cfg.model.diffusion, collate([x]), 'cpu')[0]
            raw = PoseTransform.to_raw(t)
            if init_pose_override is not None:
                raw = torch.zeros_like(raw) + torch.randn_like(raw)*0.1
            raw = raw.numpy()

            to_jax(DiffusionV2.energy_raw)(raw, *extra_args)
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
