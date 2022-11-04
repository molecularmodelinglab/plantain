
import os
import numpy as np
# from vina import Vina
from meeko import MoleculePreparation, PDBQTMolecule
import rdkit
from rdkit import Chem
from rdkit.Chem import rdMolTransforms
from rdkit.Chem.rdShapeHelpers import ComputeConfBox, ComputeUnionBox
from traceback import print_exc
import subprocess
from functools import partial
from multiprocessing import Pool
from glob import glob
import pandas as pd
import yaml
from tqdm import tqdm

from common.cfg_utils import get_config

def move_lig_to_center(lig, center):
    center_pt = rdkit.Geometry.rdGeometry.Point3D(*center)
    conf = lig.GetConformer()
    lig_center = rdMolTransforms.ComputeCentroid(conf)
    for i in range(lig.GetNumAtoms()):
        og_pos = conf.GetAtomPosition(i)
        new_pos = og_pos + center_pt - lig_center
        conf.SetAtomPosition(i, new_pos)

def get_lig_size(lig, padding=3):
    bounds = ComputeConfBox(lig.GetConformer(0))
    bounds_min = np.array([bounds[0].x, bounds[0].y, bounds[0].z])
    bounds_max = np.array([bounds[1].x, bounds[1].y, bounds[1].z])
    center = 0.5*(bounds_min + bounds_max)
    size = (bounds_max - center + padding)*2
    return tuple(center), tuple(size)

def vina_score_cached(cfg, i, row, lig_file, rec_file, exhaust=8):
    
    name = rec_file.split("/")[-1] + "_" + lig_file.split("/")[-1]
    center = (row.pocket_center_x, row.pocket_center_y, row.pocket_center_z)
    size = (row.pocket_size_x, row.pocket_size_y, row.pocket_size_z)

    lig = next(Chem.SDMolSupplier(lig_file, sanitize=True))
    move_lig_to_center(lig, center)
    _, lig_size = get_lig_size(lig)
    size = max(lig_size, size)


    preparator = MoleculePreparation(hydrate=False)
    preparator.prepare(lig)
    os.makedirs(cfg["cache_folder"], exist_ok=True)
    lig_pdbqt = cfg["cache_folder"] + "/" + name + "_lig.pdbqt"
    preparator.write_pdbqt_file(lig_pdbqt)

    out_folder = cfg["docked_folder"] + "/val_screens/" + row.pocket
    out_file = out_folder + f"/{i}.pdbqt"
    os.makedirs(out_folder, exist_ok=True)

    cmd = [cfg["vina_exec"], "--receptor", rec_file, "--ligand", lig_pdbqt, "--exhaustiveness", str(exhaust), "--cpu", str(exhaust) ]
    for c, s, ax in zip(center, size, ["x", "y", "z"]):
        cmd += ["--center_"+ax, str(c)]
        cmd += ["--size_"+ax, str(s)]
    cmd += [ "--out", out_file ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    # proc.check_returncode()
    out, err = proc.communicate()
    return out_file


def get_vina_score(cfg, tup):
    i, row = tup

    lig_file = cfg["bigbind_folder"] + "/" + row.lig_file
    rec_file = cfg["docked_folder"] + "/" + row.ex_rec_file.split(".")[0] + ".pdbqt"
    
    name = rec_file.split("/")[-1] + "_" + lig_file.split("/")[-1]

    if not os.path.exists(lig_file) or not os.path.exists(rec_file):
        return None
    
    ret = None
    try:
        ret = vina_score_cached(cfg, i, row, lig_file, rec_file)
    except KeyboardInterrupt:
        raise
    except:
        raise
        print_exc()
    
    return ret

if __name__ == "__main__":

    cfg = get_config()
    os.makedirs(cfg.platform.bigbind_vina_dir, exist_ok=True)

    for split in [ "train", "val", "test" ]:
        screen_csv = cfg.platform.bigbind_dir + f"/activities_sna_1_{split}.csv"
        screen_df = pd.read_csv(screen_csv)
        print(screen_df)
        continue

        with Pool(processes=8) as p:
            score_fn = partial(get_vina_score, cfg)
            for ret_file in p.imap_unordered(score_fn, screen_df.iterrows()):
                print(ret_file)
