
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

def prepare_rec(cfg, rec_file):
    """ Use ADFR to produce a pdbqt file from the receptor pdb file """
    rec = cfg.platform.bigbind_dir + "/" + rec_file
    rec_folder, imm_rec_file = rec.split("/")[-2:]
    out_folder = cfg.platform.bigbind_vina_dir + "/" + rec_folder
    out_file = out_folder + "/" + imm_rec_file + "qt"
    
    os.makedirs(out_folder, exist_ok=True)

    if os.path.exists(out_file): return out_file
    
    prep_cmd = f"{cfg.platform.adfr_dir}/bin/prepare_receptor"
    proc = subprocess.run([prep_cmd, "-r", rec, "-o", out_file])
    try:
        proc.check_returncode()
        return out_file
    except KeyboardInterrupt:
        raise
    except:
        print_exc()
        return None

def move_lig_to_center(lig, center):
    """ Ensure the ligand's conformer centroid is at center """
    center_pt = rdkit.Geometry.rdGeometry.Point3D(*center)
    conf = lig.GetConformer()
    lig_center = rdMolTransforms.ComputeCentroid(conf)
    for i in range(lig.GetNumAtoms()):
        og_pos = conf.GetAtomPosition(i)
        new_pos = og_pos + center_pt - lig_center
        conf.SetAtomPosition(i, new_pos)

def get_lig_size(lig, padding=3):
    """ Returns center and size of the molecule conformer """
    bounds = ComputeConfBox(lig.GetConformer(0))
    bounds_min = np.array([bounds[0].x, bounds[0].y, bounds[0].z])
    bounds_max = np.array([bounds[1].x, bounds[1].y, bounds[1].z])
    center = 0.5*(bounds_min + bounds_max)
    size = (bounds_max - center + padding)*2
    return tuple(center), tuple(size)

def run_vina(cfg, out_folder, i, row, lig_file, rec_file, exhaust=16):
    
    name = rec_file.split("/")[-1] + "_" + lig_file.split("/")[-1]
    center = (row.pocket_center_x, row.pocket_center_y, row.pocket_center_z)
    size = (row.pocket_size_x, row.pocket_size_y, row.pocket_size_z)

    cache_folder = cfg.platform.cache_dir + "/run_vina/"
    lig_pdbqt = cache_folder + name + "_lig.pdbqt"
    if not os.path.exists(lig_pdbqt):
        lig = next(Chem.SDMolSupplier(lig_file, sanitize=True))
        move_lig_to_center(lig, center)
        _, lig_size = get_lig_size(lig)
        size = max(lig_size, size)

        preparator = MoleculePreparation(hydrate=False)
        preparator.prepare(lig)
        os.makedirs(cache_folder, exist_ok=True)
        preparator.write_pdbqt_file(lig_pdbqt)

    out_file = out_folder + f"/{i}.pdbqt"
    if os.path.exists(out_file):
        return out_file

    cmd = [cfg.platform.vina_exec, "--receptor", rec_file, "--ligand", lig_pdbqt, "--exhaustiveness", str(exhaust), "--cpu", str(exhaust) ]
    for c, s, ax in zip(center, size, ["x", "y", "z"]):
        cmd += ["--center_"+ax, str(c)]
        cmd += ["--size_"+ax, str(s)]
    cmd += [ "--out", out_file ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    # proc.check_returncode()
    out, err = proc.communicate()
    return out_file


def get_vina_score(cfg, out_folder, tup):
    i, row = tup

    lig_file = cfg.platform.bigbind_dir + "/" + row.lig_file
    rec_file = prepare_rec(cfg, row.ex_rec_file)
    if rec_file is None: return None

    # if not os.path.exists(lig_file) or not os.path.exists(rec_file):
    #     return None
    
    ret = None
    try:
        ret = run_vina(cfg, out_folder, i, row, lig_file, rec_file)
    except KeyboardInterrupt:
        raise
    except:
        # raise
        print_exc()
    
    return ret

if __name__ == "__main__":

    cfg = get_config()

    for split in [ "train", "val", "test" ]:
        out_folder = cfg.platform.bigbind_vina_dir + "/" + split
        os.makedirs(out_folder, exist_ok=True)

        screen_csv = cfg.platform.bigbind_dir + f"/activities_sna_1_{split}.csv"
        screen_df = pd.read_csv(screen_csv)

        with Pool(processes=cfg.platform.vina_processes) as p:
            score_fn = partial(get_vina_score, cfg, out_folder)
            for ret_file in p.imap_unordered(score_fn, screen_df.iterrows()):
                print(ret_file)
