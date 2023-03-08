
from dataclassy import dataclass
import torch
from typing import Any, Callable, List
from common.pose_transform import MultiPose, Pose
from common.torsion import TorsionData
from data_formats.graphs.mol_graph import MolGraph, get_mol_coords
from data_formats.graphs.prot_graph import ProtGraph, get_full_rec_data

from terrace import Batch, DFRow
from terrace.batch import Batchable

@dataclass
class Transform:
    all_transforms = {}

    in_features: List[str]
    out_feature: str
    apply: Callable

    def __call__(self, cfg, x: DFRow) -> Any:
        return self.apply(cfg, x)

    @staticmethod
    def apply_many(cfg, transform_names, x: DFRow) -> DFRow:
        out = x.asdict()
        for name in transform_names:
            out[name] = Transform.all_transforms[name](cfg, x)
        return DFRow(**out)

    def get_required_features(transform_names):
        ret = set()
        for name in transform_names:
            for feat in Transform.all_transforms[name].in_features:
                ret.add(feat)
        return ret

def transform(in_features):
    def impl(func):
        ret = Transform(in_features, func.__name__, func)
        Transform.all_transforms[func.__name__] = ret
        return ret
    return impl

@transform(["lig"])
def lig_graph(cfg, x):
    return MolGraph(cfg, x.lig, None)

@transform(["rec"])
def rec_graph(cfg, x):
    return ProtGraph(cfg, x.rec)

@transform(["lig"])
def lig_embed_pose(cfg, x):
    return Pose(get_mol_coords(x.lig, 0))

@transform(["lig"])
def lig_crystal_pose(cfg, x):
    return Pose(get_mol_coords(x.lig, 0))

def get_docked_conformers(cfg, lig):
    sample = cfg.data.pose_sample
    n_confs = lig.GetNumConformers()
    num_poses = cfg.data.get("num_poses", None)
    if sample == 'all':
        assert num_poses is None
        return range(n_confs)
    elif sample == "best":
        ret = []
        for n in range(num_poses):
            ret.append(min(n,n_confs-1))
        return ret
    elif sample == 'best_and_worst':
        n_poses = 2 if num_poses is None else num_poses
        ret = []
        for n in range(n_poses-1):
            ret.append(min(n,n_confs-1))
        return ret + [n_confs - 1]
    elif sample == 'worst_and_best':
        # assert num_poses is None
        return [n_confs - 1, 0]

@transform(["lig"])
def lig_docked_poses(cfg, x):
    confs = get_docked_conformers(cfg, x.lig)
    coords = [ get_mol_coords(x.lig, c) for c in confs ]
    return MultiPose(torch.stack(coords))

@transform(["rec"])
def full_rec_data(cfg, x):
    return get_full_rec_data(cfg, x.rec)

@transform(["lig"])
def lig_torsion_data(cfg, x):
    return TorsionData.from_mol(x.lig)


