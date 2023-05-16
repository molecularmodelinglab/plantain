from copy import deepcopy
import os
import torch
from rdkit import Chem
from tqdm import tqdm
from common.pose_transform import add_pose_to_mol
from common.wandb_utils import get_old_model
from common.cfg_utils import get_config
from datasets.bigbind_struct import BigBindStructDataset
from models.diffusion_v3 import DiffusionV3
from terrace.batch import collate


def main(cfg):

    dataset = BigBindStructDataset(cfg, "val", ['lig_embed_pose', 'lig_torsion_data', 'lig_graph', 'rec_graph', 'full_rec_data'])

    # DiffusionV3 model needs to be renamed to reflect the
    # fact that is no longer a diffusion model
    model = DiffusionV3(cfg)

    # because the submodules are lazily created, we must run this model
    # before loading the state dict
    x, y = dataset[0]
    model.get_hidden_feat(collate([x]))

    model.force_field.load_state_dict(torch.load("data/plantain_ff.pt"))

    # this dataset also needs to be renamed -- really just the
    # CrossDocked dataset with bigbind splits

    for item in tqdm(dataset):
        x, y = collate([item])
        model.infer_bfgs(x)

if __name__ == "__main__":
    cfg = get_config("diffusion_v3")
    main(cfg)
