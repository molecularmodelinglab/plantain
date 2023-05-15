from copy import deepcopy
import os
import torch
from rdkit import Chem
from tqdm import tqdm
from common.pose_transform import add_pose_to_mol
from common.wandb_utils import get_old_model
from common.cfg_utils import get_config
from datasets.bigbind_struct import BigBindStructDataset
from models.diffusion_v3 import DiffusionV3, comp_f
from terrace.batch import collate


def main(cfg):
    model = get_old_model(cfg, "per_atom_energy", "latest")
    dataset = BigBindStructDataset(cfg, "val", model.get_input_feats())

    m3 = DiffusionV3(cfg)
    m3.force_field = model.force_field

    ligs = []

    def callback(x, poses):
        lig = x[0].lig
        poses = poses[0]
        cur_ligs = []
        for p in range(poses.coord.shape[0]):
            pose = poses.get(p)
            cur_lig = deepcopy(lig)
            add_pose_to_mol(cur_lig, pose)
            cur_ligs.append(cur_lig)
        ligs.append(cur_ligs)

    out_dir = "./outputs/animations/"
    os.makedirs(out_dir, exist_ok=True)
    idx = 150

    x, y = collate([dataset[idx]])
    pred_poses = m3.infer_bfgs(x, pose_callback=callback)

    for i, cur_ligs in enumerate(ligs):
        for j, lig in enumerate(cur_ligs):
            out_file = f"{out_dir}/pred_{idx}_{i}_{j}.sdf"
            writer = Chem.SDWriter(out_file)
            writer.write(lig)

if __name__ == "__main__":
    cfg = get_config("diffusion_v3")
    main(cfg)
