
import os
import numpy as np
# from vina import Vina
from meeko import MoleculePreparation, PDBQTMolecule
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolTransforms
from rdkit.Chem.rdShapeHelpers import ComputeConfBox, ComputeUnionBox
from traceback import print_exc
import subprocess
from functools import partial
from multiprocessing import Pool
from glob import glob
import pandas as pd
import yaml
import tarfile
from tqdm import tqdm

from common.cfg_utils import get_config
from common.utils import get_mol_from_file, get_prot_from_file

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

def run_vina(cfg, program, out_folder, i, row, lig_file, rec_file, exhaust=16):
    
    name = rec_file.split("/")[-1] + "_" + lig_file.split("/")[-1]
    center = (row.pocket_center_x, row.pocket_center_y, row.pocket_center_z)
    size = (row.pocket_size_x, row.pocket_size_y, row.pocket_size_z)

    cache_folder = cfg.platform.cache_dir + "/run_vina/"
    lig_pdbqt = cache_folder + name + "_lig.pdbqt"
    if not os.path.exists(lig_pdbqt) or True:
        lig = next(Chem.SDMolSupplier(lig_file, sanitize=True))

        # this is needed to dock stuff from structures_*.csv
        lig = Chem.AddHs(lig)
        AllChem.EmbedMolecule(lig)
        AllChem.UFFOptimizeMolecule(lig, 500)

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

    if program == "gnina":
        cmd = [ "apptainer", "run", "--bind", cfg.platform.bigbind_dir+","+cfg.platform.bigbind_vina_dir+","+cfg.platform.bigbind_gnina_dir+","+cfg.platform.cache_dir, cfg.platform.gnina_sif, "gnina", "--cnn", "crossdock_default2018" ]
    elif program == "vina":
        cmd = [ cfg.platform.vina_exec ]
    else:
        raise AssertionError()

    cmd += [ "--receptor", rec_file, "--ligand", lig_pdbqt, "--exhaustiveness", str(exhaust), "--cpu", str(exhaust) ]
    for c, s, ax in zip(center, size, ["x", "y", "z"]):
        cmd += ["--center_"+ax, str(c)]
        cmd += ["--size_"+ax, str(s)]
    cmd += [ "--out", out_file ]

    print(" ".join(cmd))

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    # proc.check_returncode()
    out, err = proc.communicate()
    return out_file


def get_vina_score(cfg, program, out_folder, tup):
    i, row = tup

    lig_file = cfg.platform.bigbind_dir + "/" + row.lig_file
    rec_file = prepare_rec(cfg, row.ex_rec_file)
    if rec_file is None: return None

    # if not os.path.exists(lig_file) or not os.path.exists(rec_file):
    #     return None
    
    ret = None
    try:
        ret = run_vina(cfg, program, out_folder, i, row, lig_file, rec_file)
    except KeyboardInterrupt:
        raise
    except:
        # raise
        print_exc()
    
    return ret

def dock_all(cfg, program, file_prefix):
    """ to be run in parallel on slurm """
    for split in [ "train", "val", "test" ]:
        out_folder = cfg.platform[f"bigbind_{program}_dir"]+ "/" + file_prefix + "_" + split
        os.makedirs(out_folder, exist_ok=True)

        screen_csv = cfg.platform.bigbind_dir + f"/{file_prefix}_{split}.csv"
        screen_df = pd.read_csv(screen_csv)
        seed = int(os.environ["SLURM_JOB_ID"])  if "SLURM_JOB_ID" in os.environ else 42
        screen_df = screen_df.sample(frac=1, random_state=seed)

        with Pool(processes=cfg.platform.vina_processes) as p:
            score_fn = partial(get_vina_score, cfg, program, out_folder)
            for ret_file in p.imap_unordered(score_fn, screen_df.iterrows()):
                print(ret_file)

def can_load_docked_file(cfg, program, file_prefix, split, item):
    i, row = item
    docked_file = f"{file_prefix}_{split}/{i}.pdbqt"
    full_docked_file = cfg.platform[f"bigbind_{program}_dir"]+ "/" + docked_file
    if os.path.exists(full_docked_file):
        try:
            get_mol_from_file(full_docked_file)
            return docked_file
        except KeyboardInterrupt:
            raise
        except:
            pass
            # print(f"Error processing {docked_file}")
            # print_exc()
    return None


def finalize_bigbind_vina(cfg, program, file_prefix):
    for split in [ "val", "test", "train" ]:
        screen_csv = cfg.platform.bigbind_dir + f"/{file_prefix}_{split}.csv"
        screen_df = pd.read_csv(screen_csv)

        docked_lig_files = []
        with Pool(processes=16) as p:
            f = partial(can_load_docked_file, cfg, program, file_prefix, split)
            for res in tqdm(p.imap(f, screen_df.iterrows()), total=len(screen_df)):
                docked_lig_files.append(res)

        screen_df["docked_lig_file"] = docked_lig_files
        screen_df = screen_df.dropna().reset_index(drop=True)
        
        out_file = cfg.platform[f"bigbind_{program}_dir"] + f"/{file_prefix}_{split}.csv"
        print(f"Saving docked df to {out_file}")
        screen_df.to_csv(out_file, index=False)

def make_activity_csvs(cfg, program):
    for split in ["val", "test", "train"]:
        sna_csv = cfg.platform[f"bigbind_{program}_dir"] + f"/activities_sna_1_{split}.csv"
        sna_df = pd.read_csv(sna_csv)

        act_csv = cfg.platform.bigbind_dir + f"/activities_{split}.csv"
        act_df = pd.read_csv(act_csv)
        sna_rows = {}
        for i, row in tqdm(sna_df.iterrows(), total=len(sna_df)):
            sna_rows[(row.lig_file, row.pocket)] = row
        docked_lig_files = []
        for i, act_row in tqdm(act_df.iterrows(), total=len(act_df)):
            key = (act_row.lig_file, act_row.pocket)
            if key in sna_rows:
                docked_lig_files.append(sna_rows[key].docked_lig_file)
            else:
                docked_lig_files.append("None")
        act_df["docked_lig_file"] = docked_lig_files
        len(act_df), len(sna_df)
        act_df = act_df.query("docked_lig_file != 'None'").reset_index(drop=True)

        out_csv = cfg.platform[f"bigbind_{program}_dir"] + f"/activities_{split}.csv"
        act_df.to_csv(out_csv)

if __name__ == "__main__":

    cfg = get_config("vina_ff")
    # dock_all(cfg, "gnina", "activities_sna_1")
    # dock_all(cfg, "gnina", "structures")
    finalize_bigbind_vina(cfg, "gnina", "structures")
    finalize_bigbind_vina(cfg, "gnina", "activities_sna_1")
    make_activity_csvs(cfg, "gnina")
