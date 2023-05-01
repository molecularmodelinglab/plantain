import os
import shutil
import pickle
import sys
from rdkit import Chem
import torch
from tqdm import tqdm
from common.cfg_utils import get_config
from common.pose_transform import MultiPose, add_pose_to_mol
from common.wandb_utils import get_old_model
from datasets.bigbind_struct import BigBindStructDataset
from datasets.make_dataset import make_dataset
from models.gnina import GninaPose
from models.sym_diffusion import SymDiffusion
from terrace.batch import Batch
from terrace.dataframe import DFRow
from validation.metrics import get_metrics
from validation.validate import get_preds, validate
from common.utils import flatten_dict, get_mol_from_file

def eval_combo(cfg, num_preds, shuffle_val):

    device = "cpu"
    cfg.data.num_poses = 9
    gnina = GninaPose(cfg)
    twist = get_old_model(cfg, "twist_thicc", "best_k")

    x, y, gnina_pred = get_preds(cfg, gnina, "bigbind_struct", "val", num_preds, shuffle_val)
    *_, twist_pred = get_preds(cfg, twist, "bigbind_struct", "val", num_preds, shuffle_val)

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


def main(name):
    num_preds = None
    shuffle_val = False
    dataset_name = "bigbind_struct"

    cfg = get_config("diffusion_v2")

    if name == "gnina":
        cfg.data.num_poses = 9
        model = GninaPose(cfg)
    elif name == "combo":
        return eval_combo(cfg, num_preds, shuffle_val)
    else:
        model = get_old_model(cfg, name, "best_k")
        cfg = model.cfg

    metrics, plots = validate(cfg, model, dataset_name, "val", num_preds, shuffle_val)
    for key, val in flatten_dict(metrics).items():
        print(f"{key}: {val:.3f}")

    dataset = make_dataset(cfg, dataset_name, "val", [])
    out_folder = f"outputs/pose_preds/{model.cache_key}/"

    shutil.rmtree(out_folder, ignore_errors=True)
    os.makedirs(out_folder, exist_ok=True)

    x, y, p = get_preds(cfg, model, dataset_name, "val", num_preds, shuffle_val)
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
    main(sys.argv[1])