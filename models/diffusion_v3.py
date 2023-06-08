from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from functools import lru_cache
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import dgl.backend as dF
import jax
from common.pose_transform import MultiPose, PoseTransform, Pose
from common.torsion import TorsionData
from common.jorch import to_jax
from configs.twist_score import TwistScore
from models.twister_v2 import TrueEnergy, TwistFFCoef, TwistForceField
from terrace.batch import Batch, Batchable, collate
from terrace.dataframe import DFRow, merge
from .model import Model
from .diffusion import get_transform_rmsds
from validation.metrics import get_rmsds
from multiprocessing import Pool
import multiprocessing as mp

@lru_cache
def maybe_compile(f, cfg):
    if cfg.platform.get("compile", True):
        return torch.compile(dynamic=True)(f)
    return f

# torch._dynamo.config.cache_size_limit = 2048

# from torch._dynamo.utils import CompileProfiler
# prof = CompileProfiler()
# torch._dynamo.reset()

# so far the best I can do is compile applying transformations (that is,
# translations, rotations, and torsional updates)to the ligand structure
# we need dynamic=True when compiling everything because coordinates
# and angles all have different shapes
# @torch.compile(dynamic=True, backend=prof)
def apply_transform(transforms, init_pose, lig_torsion_data):
    return transforms.apply(init_pose, lig_torsion_data)

class DiffusionV3(nn.Module, Model):
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if "score" in self.cfg.model:
            self.force_field = TwistScore(cfg.model)
        else:
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
                   inference=False):
        energy = self.force_field.get_energy(batch,
                                            hid_feat,
                                            lig_poses,
                                            inference)
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

        if hasattr(y, "lig_crystal_pose"):
            true_pose = y.lig_crystal_pose
        else:
            true_pose = collate([Pose(p.coord[0]) for p in batch.lig_docked_poses])

        transform = self.get_diffused_transforms(batch, device)
        diff_pose = transform.apply(true_pose, batch.lig_torsion_data)

        # per_atom_energy = self.cfg.model.diffusion.pred == "atom_dist"
        energy = self.get_energy(batch,
                                 hid_feat,
                                 diff_pose)
        

        diff_coord = torch.cat(diff_pose.coord, 1)
        true_coord = torch.cat(true_pose.coord, 0).unsqueeze(0)
        dists = torch.linalg.norm(diff_coord - true_coord, dim=-1)
        dists = dF.pad_packed_tensor(dists.transpose(0,1), batch.lig_graph.dgl().batch_num_nodes(), 0.0)

        rmsds = get_transform_rmsds(batch, true_pose, transform)
        noise = torch.linspace(0,1,rmsds.shape[1], device=rmsds.device).unsqueeze(0).repeat(rmsds.shape[0], 1)

        true_energy = Batch(TrueEnergy, dist=dists, noise=noise, rmsd=rmsds)

        return energy, true_energy

    def forward(self, batch, hid_feat=None):
        assert self.cfg.model.inference.optimizer == "bfgs"
        lig_pose, energy = self.infer_bfgs(batch, hid_feat)
        return Batch(DFRow, lig_pose=lig_pose, energy=energy)

    def predict_train(self, x, y, task_names, split, batch_idx):
        hid_feat = self.get_hidden_feat(x)
        pred_energy, true_energy = self.diffuse_energy(x, y, hid_feat)
        ret_dif = Batch(DFRow, pred_energy=pred_energy, true_energy=true_energy)
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
            ret = merge([ret_dif, ret_pred])
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
            # explanation, *rest = torch._dynamo.explain(DiffusionV3.infer_bfgs_single, self, x0, hf0, pose_callback)
            # print(explanation)
            # print(rest[-1])
            # exit()
            ret.append(self.infer_bfgs_single(x0, hf0, pose_callback))
        return collate(ret)
    
    # @torch.compile(dynamic=True)#, backend=prof)
    def get_inference_energy(self, x, hid_feat, init_pose, transforms):
        """ This is the function we want to differentiate -- get the
        predicted 'energy' from the ligand translation, rotation,
        and torsional angles """
        
        # explanation, *rest = torch._dynamo.explain(transforms.apply, init_pose, x.lig_torsion_data)
        # print(explanation)
        # print(rest[-1])
        # exit()
        
        poses = apply_transform(transforms, init_pose, x.lig_torsion_data)
        # poses = transforms.apply(init_pose, x.lig_torsion_data)
        Us = self.get_energy(x, hid_feat, poses, True).energy
        # U = Us.sum()
        return Us

    # @torch.compile(dynamic=True, backend=prof)
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
        

        # od = torch.jit.trace_module(self, {"get_raw_inference_energy": (params[0], params[1], [params[2]])})
        
        def closure():
            optimizer.zero_grad()
            if pose_callback is not None:
                pose_callback(x, get_poses())
            params = optimizer.param_groups[0]["params"]
            transforms = Batch(PoseTransform, trans=params[0], rot=params[1], tor_angles=[params[2]])
            
            # because terrace lazily initializes modules, we need to call the
            # score function once before compiling
            if hasattr(self.force_field, "fast_score") and not self.force_field.fast_score.is_initialized():
                U = self.get_inference_energy(x, hid_feat, init_pose, transforms).sum()
            else:
                U = maybe_compile(self.get_inference_energy, self.cfg)(x, hid_feat, init_pose, transforms).sum()
            U.backward()
            # U = self.infer_backward(x, hid_feat, init_pose, transforms)
            # print(U)
            # torch._dynamo.explain(DiffusionV3.infer_backward, self, x, hid_feat, init_pose, transforms)
            return U
        
        optimizer.step(closure)

        params = optimizer.param_groups[0]["params"]
        transforms = Batch(PoseTransform, trans=params[0], rot=params[1], tor_angles=[params[2]])
        Us = self.get_inference_energy(x, hid_feat, init_pose, transforms)[0]

        # sort poses according to increasing "energy"
        idx = torch.argsort(Us)
        energy = Us[idx]

        coord = get_poses().coord[0]
        poses = MultiPose(coord[idx])

        return poses, energy