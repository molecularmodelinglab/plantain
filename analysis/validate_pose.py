import os
import shutil
import pickle
import sys
import warnings
import pandas as pd
from rdkit import Chem
import torch
from tqdm import tqdm
from common.cfg_utils import get_config
from common.pose_transform import MultiPose, add_multi_pose_to_mol, add_pose_to_mol
from common.wandb_utils import get_old_model
from datasets.crossdocked import CrossDockedDataset
from datasets.make_dataset import make_dataset
from models.diffdock import DiffDock, get_diffdock_indexes
from models.diffusion_v3 import DiffusionV3
from models.gnina import GninaPose
from models.gnina_combo import GninaComboPose
from models.vina import VinaPose
from terrace.batch import Batch, collate
from terrace.dataframe import DFRow
from validation.metrics import get_metrics
from validation.validate import get_preds, validate
from common.utils import flatten_dict, get_mol_from_file

def eval_naive_combo(cfg, dataset_name, num_preds, split, shuffle_val):

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
    v2_to_v3 = False
    num_preds = 2
    shuffle_val = False
    timing = True
    dataset_name = "crossdocked"
    # dataset_name = "bigbind_struct"
    subset = None

    cfg = get_config("icml_v3")

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
        cfg.data.num_poses = 16
        model = GninaComboPose(cfg, "wandb:ybslp3tt:v11")
    elif name == "naive_combo":
        return eval_naive_combo(cfg, dataset_name, num_preds, split, shuffle_val)
    else:
        if v2_to_v3:
            assert name == "thin_chungus" and tag == "best_k"
            cfg = get_config("cur_best_v3")
            model = DiffusionV3(cfg)
            model.cache_key = "thin_chungus_v3"

            dataset = CrossDockedDataset(cfg, "val", ['lig_embed_pose', 'lig_torsion_data', 'lig_graph', 'rec_graph', 'full_rec_data'])

            x, y = dataset[0]
            model.get_hidden_feat(collate([x]))

            model.force_field.load_state_dict(torch.load("data/cur_best.pt"))
        else:
            model = get_old_model(cfg, name, tag)
            model.cfg.model.score = cfg.model.score
            model.force_field.cfg.score = cfg.model.score
            cfg = model.cfg

    cfg.batch_size = 4
    del cfg.batch_sampler
    cfg.model.diffusion.only_pred_local_min = True

    prefix = "" if subset is None else subset
    metrics, (x, y, p, runtimes) = validate(cfg, model, dataset_name, split, num_preds, shuffle_val, subset_indexes, timing)
    for key, val in flatten_dict(metrics).items():
        print(f"{prefix}_{key}: {val:.3f}")

    print(f"Mean runtime: {sum(runtimes)/len(runtimes)}")

    if subset is not None: return
    # return

    dataset = make_dataset(cfg, dataset_name, split, [])
    out_folder = f"outputs/pose_preds/{model.cache_key}/"

    shutil.rmtree(out_folder, ignore_errors=True)
    os.makedirs(out_folder, exist_ok=True)

    # x, y, p = get_preds(cfg, model, dataset_name, split, num_preds, shuffle_val)

    lig_files = []
    rec_files = []
    pred_files = []
    for i, (lig_file, poc_file, p_pose) in enumerate(zip(x.lig_crystal_file, x.rec_file, tqdm(p.lig_pose))):
        lf = lig_file
        rf = poc_file
        # if dataset_name == "bigbind_struct":
        #     lf = cfg.platform.bigbind_dir + "/" + lig_file
        #     rf = cfg.platform.bigbind_dir + "/" + poc_file
        # elif dataset_name == "crossdocked":
        #     lf = cfg.platform.crossdocked_dir + "/" + lig_file
        #     rf = cfg.platform.crossdocked_dir + "/" + poc_file
        # else:
        #     raise AssertionError
        mol = get_mol_from_file(lf)
        add_multi_pose_to_mol(mol, p_pose)
        pose_file = out_folder + str(i) + ".sdf"
        writer = Chem.SDWriter(pose_file)
        for c in range(mol.GetNumConformers()):
            writer.write(mol, c)
        writer.close()
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main(sys.argv[1], "test", tag)