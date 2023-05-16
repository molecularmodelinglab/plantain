from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
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
from .diffusion_v2 import DiffPred
from validation.metrics import get_rmsds
from multiprocessing import Pool
import multiprocessing as mp

# so far the best I can do is compile applying transformations (that is,
# translations, rotations, and torsional updates)to the ligand structure
# we need dynamic=True when compiling everything because coordinates
# and angles all have different shapes
# @torch.compile(dynamic=True)
def apply_transform(transforms, init_pose, lig_torsion_data):
    return transforms.apply(init_pose, lig_torsion_data)

class DiffusionV3(nn.Module, Model):
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.force_field = TwistForceField(cfg.model)

    @staticmethod
    def get_name() -> str:
        return "diffusion_v3"

    def get_input_feats(self):
        ret = ["lig_embed_pose", "lig_torsion_data"]
        ret += self.force_field.get_input_feats()
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

    def get_diffused_transforms(self, batch, device, timesteps=None):
        diff_cfg = self.cfg.model.diffusion
        if timesteps is None:
            timesteps = diff_cfg.timesteps

        return PoseTransform.make_diffused(diff_cfg, timesteps, batch, device)

    def diffuse_energy(self, batch, y, hid_feat=None):
        
        if hid_feat is None:
            hid_feat= self.get_hidden_feat(batch)
        device = batch.lig_graph.ndata.cat_feat.device

        true_pose = y.lig_crystal_pose

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

    def forward(self, batch, hid_feat=None):
        assert self.cfg.model.inference.optimizer == "bfgs"
        lig_pose, energy = self.infer_bfgs(batch, hid_feat)
        return Batch(DFRow, lig_pose=lig_pose, energy=energy)

    def predict_train(self, x, y, task_names, split, batch_idx):
        hid_feat = self.get_hidden_feat(x)
        diff_energy, diff_rmsds, inv_dist_mat = self.diffuse_energy(x, y, hid_feat)
        ret_dif = Batch(DiffPred, diffused_energy=diff_energy, diffused_rmsds=diff_rmsds, inv_dist_mat=inv_dist_mat, hid_feat=hid_feat)
        if "predict_lig_pose" in task_names and (split != "train" or batch_idx % self.cfg.metric_reset_interval == 0):
            with torch.no_grad():
                ret_pred = self(x, hid_feat)
            ret = merge([ret_dif, ret_pred])
            ret._batch_type = DiffPred
            return ret
        else:
            return ret_dif

    @staticmethod
    def get_init_pose(x):
        init_poses = []
        for x0 in x:
            embed_pose = x0.lig_embed_pose
            rec_mean = x0.full_rec_data.ndata.coord.mean(0)
            lig_mean = embed_pose.coord.mean(0)
            init_poses.append(Pose(coord=embed_pose.coord + rec_mean - lig_mean))
        return collate(init_poses)

    @torch.no_grad()
    def infer_bfgs(self, x, hid_feat = None, pose_callback=None):
        if hid_feat is None:
            hid_feat = self.get_hidden_feat(x)
        ret = []
        for i, (L, Rf) in enumerate(zip(x.lig_graph.dgl().batch_num_nodes(),
                                x.full_rec_data.dgl().batch_num_nodes())):
            x0 = collate([x[i]], lazy=True)
            hf0 = Batch(DFRow,
                l_rf_coef=hid_feat.l_rf_coef.detach()[i:(i+1),:L,:Rf],
                ll_coef=hid_feat.ll_coef.detach()[i:(i+1),:L,:L]
            )
            ret.append(self.infer_bfgs_single(x0, hf0, pose_callback))
        return collate(ret)
    
    def get_inference_energy(self, x, hid_feat, init_pose, transforms):
        """ This is the function we want to differentiate -- get the
        predicted 'energy' from the ligand translation, rotation,
        and torsional angles """
        poses = apply_transform(transforms, init_pose, x.lig_torsion_data)
        # poses = transforms.apply(init_pose, x.lig_torsion_data)
        Us = self.get_energy(x, hid_feat, poses, True)
        U = Us.sum()
        return U

    def infer_bfgs_single(self, x, hid_feat, pose_callback):
        """ Use Pytorch's L-BFGS optimizer to minimize the predicted score
        w/r/t the translation, rotation, and torsional angles """
        device = hid_feat.ll_coef.device
        inf_cfg = self.cfg.model.inference
        n_poses = inf_cfg.num_optim_poses
        
        rand_transforms = PoseTransform.make_initial(self.cfg.model.diffusion, x, device, n_poses)
        init_pose = self.get_init_pose(x)

        params = []
        params.append(nn.Parameter(rand_transforms.trans))
        params.append(nn.Parameter(rand_transforms.rot))
        params.append(nn.Parameter(rand_transforms.tor_angles[0]))
        optimizer = torch.optim.LBFGS(params, inf_cfg.learn_rate, inf_cfg.max_iter)
        
        def get_poses():
            params = optimizer.param_groups[0]["params"]
            transforms = Batch(PoseTransform, trans=params[0], rot=params[1], tor_angles=[params[2]])
            return transforms.apply(init_pose, x.lig_torsion_data)

        def closure():
            optimizer.zero_grad()
            if pose_callback is not None:
                pose_callback(x, get_poses())
            params = optimizer.param_groups[0]["params"]
            transforms = Batch(PoseTransform, trans=params[0], rot=params[1], tor_angles=[params[2]])
            U = self.get_inference_energy(x, hid_feat, init_pose, transforms)
            U.backward()
            return U
        
        optimizer.step(closure)

        return get_poses()[0]