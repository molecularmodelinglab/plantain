import os
import pickle
from rdkit import Chem
from tqdm import tqdm
from common.cfg_utils import get_config
from common.pose_transform import add_pose_to_mol
from common.wandb_utils import get_old_model
from datasets.bigbind_struct import BigBindStructDataset
from models.sym_diffusion import SymDiffusion
from validation.validate import get_preds, validate
from common.utils import flatten_dict, get_mol_from_file

num_preds = 2

old_cfg = get_config("diffusion_v2")
model = get_old_model(old_cfg, "twist_thicc", "best_k")
cfg = model.cfg
# model.cfg = cfg
metrics, plots = validate(cfg, model, "bigbind_struct", "val", num_preds, False)
for key, val in flatten_dict(metrics).items():
    print(f"{key}: {val:.3f}")

dataset = BigBindStructDataset(cfg, "val", [])
out_folder = f"outputs/pose_preds/{model.cache_key}/"
os.makedirs(out_folder, exist_ok=True)
x, y, p = get_preds(cfg, model, "bigbind_struct", "val", num_preds, False)
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