
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

def run_vina(cfg, out_folder, i, row, lig_file, rec_file, exhaust=16):
    
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

def dock_all(cfg, file_prefix):
    """ to be run in parallel on slurm """
    for split in [ "train", "val", "test" ]:
        out_folder = cfg.platform.bigbind_vina_dir + "/" + file_prefix + "_" + split
        os.makedirs(out_folder, exist_ok=True)

        screen_csv = cfg.platform.bigbind_dir + f"/{file_prefix}_{split}.csv"
        screen_df = pd.read_csv(screen_csv)
        seed = int(os.environ["SLURM_JOB_ID"])  if "SLURM_JOB_ID" in os.environ else 42
        screen_df = screen_df.sample(frac=1, random_state=seed)

        with Pool(processes=cfg.platform.vina_processes) as p:
            score_fn = partial(get_vina_score, cfg, out_folder)
            for ret_file in p.imap_unordered(score_fn, screen_df.iterrows()):
                print(ret_file)

def can_load_docked_file(file_prefix, split, item):
    i, row = item
    docked_file = f"{file_prefix}_{split}/{i}.pdbqt"
    full_docked_file = cfg.platform.bigbind_vina_dir + "/" + docked_file
    if os.path.exists(full_docked_file):
        try:
            get_mol_from_file(full_docked_file)
            return docked_file
        except KeyboardInterrupt:
            raise
        except:
            print(f"Error processing {docked_file}")
            print_exc()
    return None


def finalize_bigbind_vina(cfg, file_prefix):
    for split in [ "val", "test", "train" ]:
        screen_csv = cfg.platform.bigbind_dir + f"/{file_prefix}_{split}.csv"
        screen_df = pd.read_csv(screen_csv)

        docked_lig_files = []
        with Pool(processes=16) as p:
            f = partial(can_load_docked_file, file_prefix, split)
            for res in tqdm(p.imap(f, screen_df.iterrows()), total=len(screen_df)):
                docked_lig_files.append(res)

        screen_df["docked_lig_file"] = docked_lig_files
        screen_df = screen_df.dropna().reset_index(drop=True)
        
        out_file = cfg.platform.bigbind_vina_dir + f"/{file_prefix}_{split}.csv"
        print(f"Saving docked df to {out_file}")
        screen_df.to_csv(out_file, index=False)

def add_rec_file(file):
    rec_file = cfg.platform.bigbind_dir + "/" + file
    out_file = rec_file.split(".")[0] + ".mmtf"
    # if not os.path.exists(out_file):
    get_prot_from_file(rec_file)


def add_lig_file(file):
    docked_file = cfg.platform.bigbind_vina_dir + "/" + file
    out_file = docked_file.split(".")[0] + ".pkl"
    # if not os.path.exists(out_file):
    get_mol_from_file(docked_file)

def get_file_size(f):
    f.seek(0, os.SEEK_END)
    ret = f.tell()
    f.seek(0, os.SEEK_SET)
    return ret

def tar_structures(cfg, prefix, split):

    df = pd.read_csv(cfg.platform.bigbind_vina_dir + f"/{prefix}_{split}.csv")
    tar = tarfile.open(cfg.platform.bigbind_vina_dir + f"/{split}_files.tar", "w:")

    # for file in tqdm(df.ex_rec_pocket_file):
    #     add_rec_file(tar, file)
    with Pool(processes=8) as p:

        rec_files = df.ex_rec_pocket_file.unique()
        lig_files = df.docked_lig_file.unique()

        for _ in tqdm(p.imap(add_rec_file, rec_files), total=len(rec_files)):
            pass
        for _ in tqdm(p.imap(add_lig_file, lig_files), total=len(lig_files)):
            pass

        for file in rec_files:
            rec_file = cfg.platform.bigbind_dir + "/" + file
            out_file = rec_file.split(".")[0] + ".mmtf"
            with open(out_file, 'rb') as f:
                info = tarfile.TarInfo(file)
                info.size = get_file_size(f)
                tar.addfile(info, f)

        for file in lig_files:
            docked_file = cfg.platform.bigbind_vina_dir + "/" + file
            out_file = docked_file.split(".")[0] + ".pkl"
            with open(out_file, 'rb') as f:
                info = tarfile.TarInfo(file)
                info.size = get_file_size(f)
                tar.addfile(info, f)

    tar.close()

def tar_all_structures(cfg, prefix):
    for split in ("val", "test", "train"):
        print(f"Tarring {split}")
        tar_structures(cfg, prefix, split)

if __name__ == "__main__":

    cfg = get_config("vina")
    # dock_all(cfg, "activities_sna_1")
    # dock_all(cfg, "structures")
    # finalize_bigbind_vina(cfg, "structures")
    tar_all_structures(cfg, "activities_sna_1")
