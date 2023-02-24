from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import jax
from common.pose_transform import PoseTransform, Pose
from common.torsion import TorsionData
from common.jorch import to_jax
from models.sym_force_field import SymForceField
from terrace.batch import Batch, collate
from terrace.dataframe import DFRow, merge
from .model import Model
from validation.metrics import get_rmsds
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
from .diffusion import get_transform_rmsds
from scipy.optimize import minimize

# mp.set_start_method('forkserver', force=True)

class SymDiffusion(nn.Module, Model):
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.force_field = SymForceField(cfg)

    @staticmethod
    def get_name() -> str:
        return "sym_diffusion"

    def get_input_feats(self):

        ret = ["lig_embed_pose", "lig_torsion_data"] + self.force_field.get_input_feats()
        # if "pose_sample" in self.cfg.data:
        #     ret.append("lig_docked_poses")
        return ret

    def get_tasks(self):
        return ["predict_lig_pose"]
        
    def get_interaction_matrix(self, x):
        return self.force_field.get_interaction_matrix(x)

    def get_energy(self,
                   batch,
                   inter_mat,
                   batch_lig_pose,
                   batch_transform):
        lig_poses = batch_transform.apply(batch_lig_pose, batch.lig_torsion_data)
        return self.force_field.get_energy(inter_mat,
                                           batch,
                                           lig_poses)

    @staticmethod
    def energy_raw(t_raw,
                   cfg,
                   rot_edges,
                   rot_masks,
                   inter_mat,
                   init_lig_coord,
                   rec_coord,
                   weight,
                   bias):
        # tor_data = tor_data_wrap.obj
        transform = PoseTransform.from_raw(t_raw)
        tor_data = TorsionData(rot_edges, rot_masks)
        lig_pose = transform.apply(Pose(coord=init_lig_coord), tor_data)
        comb_coords = torch.cat((lig_pose.coord, rec_coord), 0)
        U = SymForceField.get_energy_single(cfg.model,
                                        inter_mat,
                                        comb_coords,
                                        weight,
                                        bias)
        return U

    def get_diffused_transforms(self, batch, device, timesteps=None):
        diff_cfg = self.cfg.model.diffusion
        if timesteps is None:
            timesteps = diff_cfg.timesteps

        return PoseTransform.make_diffused(diff_cfg, timesteps, batch, device)

    def diffuse_energy(self, batch, y, inter_mat):
        device = inter_mat[0].device
        transform = self.get_diffused_transforms(batch, device)
        
        # if we only have the ligand, make sure it's only
        # torsion angles we modify
        if not hasattr(batch, "rec"):
            transform.trans = torch.zeros_like(transform.trans)
            transform.rot = torch.zeros_like(transform.rot)

        if hasattr(y, "lig_crystal_pose"):
            true_pose = y.lig_crystal_pose
        else:
            true_pose = batch.lig_embed_pose # collate([Pose(p.coord[0]) for p in batch.lig_docked_poses])

        energy = self.get_energy(batch, 
                              inter_mat,
                              true_pose,
                              transform)
        rmsds = get_transform_rmsds(batch, true_pose, transform)
        return energy, rmsds

    def forward(self, batch, inter_mat=None):
        if not hasattr(batch, "rec"):
            return Batch(DFRow, dummy=torch.zeros(len(batch)))
        # assert hasattr(batch, "full_rec_data")

        if inter_mat is None:
            inter_mat = self.get_interaction_matrix(batch)
        lig_pose, energy = self.infer_bfgs(batch, inter_mat)
        return Batch(DFRow, lig_pose=lig_pose, energy=energy)

    def predict_train(self, x, y, task_names, split, batch_idx):
        inter_mat = self.get_interaction_matrix(x)
        assert self.cfg.model.diffusion.energy_rmsd
        diff_energy, diff_rmsds = self.diffuse_energy(x, y, inter_mat)
        ret_dif = Batch(DFRow, diffused_energy=diff_energy, diffused_rmsds=diff_rmsds)   
        if "predict_lig_pose" in task_names and (split != "train" or batch_idx % 50 == 0):
            with torch.no_grad():
                ret_pred = self(x, inter_mat)
            return merge([ret_dif, ret_pred])
        else:
            return ret_dif

    @torch.no_grad()
    def infer_bfgs(self, x, inter_mat):

        bias, weight = self.force_field.scale_output.parameters()

        x = x.cpu()
        ret = []
        args = []
        for i in range(len(x)):
            args.append((self.cfg, x[i], inter_mat[i].detach().cpu(), weight.detach().cpu(), bias.detach().cpu()))
        
        if not hasattr(SymDiffusion, "jit_infer"):
            SymDiffusion.jit_infer = jax.jit(jax.value_and_grad(to_jax(SymDiffusion.energy_raw)), static_argnums=1)

        if self.cfg.platform.infer_workers > 0:
            with ThreadPoolExecutor(max_workers=self.cfg.platform.infer_workers) as p:
                for res in p.map(SymDiffusion.infer_bfgs_single, args):
                    ret.append(res)
        else:
            for arg in args:
                ret.append(SymDiffusion.infer_bfgs_single(arg))

        return collate(ret)

    @staticmethod
    def infer_bfgs_single(args, debug=False):
        cfg, x, inter_mat, weight, bias = args
        f = SymDiffusion.jit_infer
        # f = jax.jit(jax.value_and_grad(to_jax(SymDiffusion.energy_raw)), static_argnums=1) # deepcopy(Diffusion.jit_infer)
        # f = jax.value_and_grad(to_jax(SymDiffusion.energy_raw))

        method = "BFGS"
        options = {
            # "disp": True,
            "maxiter": 1 if debug else 30,
        }

        if cfg.model.project_full_atom:
            rec_coord = x.full_rec_data.coord
        else:
            rec_coord = x.rec_graph.ndata.coord

        extra_args = (
            cfg,
            x.lig_torsion_data.rot_edges.detach().cpu().numpy(),
            x.lig_torsion_data.rot_masks.detach().cpu().numpy(),
            inter_mat.detach().cpu().numpy(),
            x.lig_embed_pose.coord.detach().cpu().numpy(),
            rec_coord.detach().cpu().numpy(),
            weight.detach().cpu().numpy(),
            bias.detach().cpu().numpy(),
        )
        best_energy = 1e10
        n_tries = cfg.model.diffusion.get("optim_tries", 1 if debug else 15)
        for i in range(n_tries):
            t = PoseTransform.make_initial(cfg.model.diffusion, collate([x]), 'cpu')[0]
            raw = PoseTransform.to_raw(t).numpy()
            res = minimize(f, raw, extra_args, method=method, jac=True, options=options)
            opt_raw = torch.tensor(res.x, dtype=torch.float32)
            t_opt = PoseTransform.from_raw(opt_raw)
            pose = t_opt.apply(x.lig_embed_pose, x.lig_torsion_data)
            if res.fun < best_energy:
                best_energy = res.fun
                best_pose = pose
        
        return best_pose, best_energy