import os
import shutil
import pickle
import sys
import pandas as pd
from rdkit import Chem
import torch
from tqdm import tqdm
from common.cfg_utils import get_config
from common.pose_transform import MultiPose, add_pose_to_mol
from common.wandb_utils import get_old_model
from datasets.bigbind_struct import BigBindStructDataset
from datasets.make_dataset import make_dataset
from models.diffdock import DiffDock, get_diffdock_indexes
from models.gnina import GninaPose
from models.sym_diffusion import SymDiffusion
from models.vina import VinaPose
from terrace.batch import Batch, collate
from terrace.dataframe import DFRow
from validation.metrics import get_metrics
from validation.validate import get_preds, validate
from common.utils import flatten_dict, get_mol_from_file

def eval_combo(cfg, dataset_name, num_preds, split, shuffle_val):

    device = "cpu"
    cfg.data.num_poses = 9
    gnina = GninaPose(cfg)
    twist = get_old_model(cfg, "even_more_tor", "best_k")

    x, y, gnina_pred = get_preds(cfg, gnina, dataset_name, split, num_preds, shuffle_val)
    *_, twist_pred = get_preds(cfg, twist, dataset_name, split, num_preds, shuffle_val)

    x = x.to(device)
    y = y.to(device)

    # concatentate their predicted poses together
    combined_coords = []
    for p1, p2 in zip(gnina_pred, twist_pred):
        combined_coords.append(torch.cat((p1.lig_pose.coord.to(device),
                                          p2.lig_pose.coord.to(device)), 0))
    comb_pred = Batch(DFRow, lig_pose=Batch(MultiPose, coord=combined_coords))

    metrics = get_metrics(cfg, ["predict_lig_pose"], offline=True).to(device)
    for metric in metrics.values():
        metric.update(x, comb_pred, y)

    comp_mets = {
        key: val.cpu().compute() for key, val in metrics.items()
    }
    for key, val in flatten_dict(comp_mets).items():
        print(f"{key}: {val:.3f}")


def main(name, split, tag):
    print(f"Evaluating {name}:{tag} on {split}")
    num_preds = None
    shuffle_val = False
    dataset_name = "crossdocked"
    # dataset_name = "bigbind_struct"
    subset = None

    cfg = get_config("diffusion_v2")

    if subset is None:
        subset_indexes = None
    elif subset == "diffdock":
        # bb_diffdock_csv = cfg.platform.diffdock_dir + "/data/bb_struct_val.csv"
        # subset_indexes = set(pd.read_csv(bb_diffdock_csv)["Unnamed: 0"])
        dataset = make_dataset(cfg, dataset_name, split, [])
        subset_indexes = get_diffdock_indexes(cfg, dataset)

    if name == "gnina":
        cfg.data.num_poses = 9
        model = GninaPose(cfg)
    elif name == "vina":
        cfg.data.num_poses = 9
        model = VinaPose(cfg)
    elif name == "diffdock":
        cfg.data.num_poses = 40
        model = DiffDock(cfg, split)
    elif name == "combo":
        return eval_combo(cfg, dataset_name, num_preds, split, shuffle_val)
    else:
        model = get_old_model(cfg, name, tag)
        cfg = model.cfg

    cfg.batch_size = 8
    cfg.model.diffusion.only_pred_local_min = True

    prefix = "" if subset is None else subset
    metrics, plots = validate(cfg, model, dataset_name, split, num_preds, shuffle_val, subset_indexes)
    for key, val in flatten_dict(metrics).items():
        print(f"{prefix}_{key}: {val:.3f}")

    if subset is not None: return
    # return

    dataset = make_dataset(cfg, dataset_name, split, [])
    out_folder = f"outputs/pose_preds/{model.cache_key}/"

    shutil.rmtree(out_folder, ignore_errors=True)
    os.makedirs(out_folder, exist_ok=True)

    x, y, p = get_preds(cfg, model, dataset_name, split, num_preds, shuffle_val, subset_indexes)

    lig_files = []
    rec_files = []
    pred_files = []
    for i, (lig_file, poc_file, p_pose) in enumerate(zip(dataset.structures.lig_file, dataset.structures.ex_rec_pocket_file, tqdm(p.lig_pose.get(0)))):
        lf = cfg.platform.bigbind_dir + "/" + lig_file
        rf = cfg.platform.bigbind_dir + "/" + poc_file
        mol = get_mol_from_file(lf)
        add_pose_to_mol(mol, p_pose)
        pose_file = out_folder + str(i) + ".sdf"
        writer = Chem.SDWriter(pose_file)
        writer.write(mol)
        pred_files.append(pose_file)
        lig_files.append(lf)
        rec_files.append(rf)

    with open(out_folder + "files.pkl", "wb") as f:
        pickle.dump((lig_files, rec_files, pred_files), f)

    print(f"Saved output pose files to '{out_folder}'")

if __name__ == "__main__":
    try:
        tag = sys.argv[2]
    except IndexError:
        tag = "best_k"
    main(sys.argv[1], "val", tag)