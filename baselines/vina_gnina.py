
import os
import time
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
from common.utils import get_mol_from_file, get_mol_from_file_no_cache, get_prot_from_file

def get_crossdocked_dir(cfg):
    return cfg.platform.crossdocked_dir

def prepare_rec(cfg, rec_file):
    """ Use OpenBabel to produce a pdbqt file from the receptor pdb file """
    rec = get_crossdocked_dir(cfg) + "/" + rec_file
    rec_folder, imm_rec_file = rec.split("/")[-2:]
    out_folder = cfg.platform.crossdocked_vina_dir + "/" + rec_folder
    out_file = out_folder + "/" + imm_rec_file + "qt"
    
    os.makedirs(out_folder, exist_ok=True)

    if os.path.exists(out_file): return out_file
    
    # prep_cmd = f"{cfg.platform.adfr_dir}/bin/prepare_receptor"
    # proc = subprocess.run([prep_cmd, "-r", rec, "-o", out_file])
    prep_cmd = [cfg.platform.obabel_exec, "-xr", "-ipdb", rec, "-opdbqt", "-O"+out_file]
    print(f"Running {' '.join(prep_cmd)}")
    proc = subprocess.run(prep_cmd)
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

# Should be zero. Make 1 to test gnina on its training set
GNINA_ITER = 0
# Don't re-run openbabel before running gnina
PREP_FOR_GNINA = False
def run_vina(cfg, program, out_folder, i, row, lig_file, rec_file, timing):
    
    name = rec_file.split("/")[-1] + "_" + lig_file.split("/")[-1]
    center = (row.pocket_center_x, row.pocket_center_y, row.pocket_center_z)
    size = (row.pocket_size_x, row.pocket_size_y, row.pocket_size_z)

    cache_folder = cfg.platform.cache_dir + "/run_vina/"
    lig_pdbqt = cache_folder + name + "_lig.pdbqt"
    if not os.path.exists(lig_pdbqt):
        lig = next(Chem.SDMolSupplier(lig_file, sanitize=True))

        # # this is needed to dock stuff from structures_*.csv
        lig = Chem.AddHs(lig, addCoords=True)
        # AllChem.EmbedMolecule(lig)
        # AllChem.UFFOptimizeMolecule(lig, 500)

        move_lig_to_center(lig, center)
        _, lig_size = get_lig_size(lig)
        size = max(lig_size, size)

        preparator = MoleculePreparation(hydrate=False)
        preparator.prepare(lig)
        os.makedirs(cache_folder, exist_ok=True)
        preparator.write_pdbqt_file(lig_pdbqt)

    out_file = out_folder + f"/{i}.pdbqt"
    if program == "gnina":
        out_file = out_folder + f"/{i}.sdf"

    if os.path.exists(out_file) and not timing:
        try:
            get_mol_from_file_no_cache(out_file)
            return out_file
        except:
            print_exc()

    if program == "gnina":
        # todo: remove --nv option when running without gpu
        if "gnina_exec" in cfg.platform:
            cmd = [ cfg.platform.gnina_exec ]
        else:
            cmd = [ "apptainer", "run", "--nv", "--bind", get_crossdocked_dir(cfg)+","+cfg.platform.crossdocked_vina_dir+","+cfg.platform.crossdocked_gnina_dir+","+cfg.platform.cache_dir, cfg.platform.gnina_sif, "gnina" ]
        if GNINA_ITER is not None:
            cmd += [ "--cnn_weights" ] + glob(f"./prior_work/gnina/*{GNINA_ITER}_iter_*.caffemodel")
            cmd += [ "--cnn_model" ] + [ "./prior_work/gnina/default2018.model" ]*5

        if not PREP_FOR_GNINA:
            rec_file = get_crossdocked_dir(cfg) + "/" + row.crossdock_rec_file
            lig_pdbqt = lig_file

    elif program == "vina":
        cmd = [ cfg.platform.vina_exec ]
    else:
        raise AssertionError()

    cmd += [ "--receptor", rec_file, "--ligand", lig_pdbqt, "--cpu", str(1 if timing else cfg.platform.vina_processes) ]
    for c, s, ax in zip(center, size, ["x", "y", "z"]):
        cmd += ["--center_"+ax, str(c)]
        cmd += ["--size_"+ax, str(s)]
    cmd += [ "--out", out_file ]

    print("Docking with", " ".join(cmd))

    cur_time = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    # proc.check_returncode()
    out, err = proc.communicate()
    runtime = time.time() - cur_time

    return out_file, runtime

def get_lig_file(row):
    return row.lig_uff_file

def get_rec_file(row):
    return row.crossdock_rec_file

def get_vina_score(cfg, program, out_folder, tup, timing):
    i, row = tup

    lig_file = get_crossdocked_dir(cfg) + "/" + get_lig_file(row)
    rec_file = prepare_rec(cfg, get_rec_file(row))
    if rec_file is None: return None

    # if not os.path.exists(lig_file) or not os.path.exists(rec_file):
    #     return None
    
    ret = "null", "null"
    try:
        ret = run_vina(cfg, program, out_folder, i, row, lig_file, rec_file, timing)
    except KeyboardInterrupt:
        raise
    except:
        # raise
        print_exc()
    
    return ret

# set timing to true if you want to time vina/gnina rigorously. This sets the available CPUs
# to 1 and disables caching, Set to false if you just want docking results
def dock_all(cfg, program, file_prefix, timing, random_sample=False):
    """ to be run in parallel on slurm """
    for split in [ "test" ]:
        out_folder = cfg.platform[f"crossdocked_{program}_dir"]+ "/" + file_prefix + "_" + split
        os.makedirs(out_folder, exist_ok=True)

        screen_csv = get_crossdocked_dir(cfg) + f"/{file_prefix}_{split}.csv"
        screen_df = pd.read_csv(screen_csv)
        seed = int(os.environ["SLURM_JOB_ID"])  if "SLURM_JOB_ID" in os.environ else 42
        screen_df = screen_df.sample(frac=1, random_state=seed)

        # with Pool(processes=cfg.platform.vina_processes) as p:
        #     score_fn = partial(get_vina_score, cfg, program, out_folder)
        #     for ret_file in p.imap_unordered(score_fn, screen_df.iterrows()):
        #         print(ret_file)
    
        if timing:
            
            # try to get previous runtimes if we crashed
            fname = f"outputs/{program}_crossdocked_{file_prefix}_{split}_runtimes.txt"
            try:
                with open(fname, "r") as f:
                    prev_lines = f.readlines()
            except FileNotFoundError:
                prev_lines = []

            f = open(fname, "w")

        for i, item in enumerate(tqdm(screen_df.iterrows(), total=len(screen_df))):
            if timing and i < len(prev_lines):
                # print("skipping", i)
                f.write(prev_lines[i])
                continue
            out_file, runtime = get_vina_score(cfg, program, out_folder, item, timing)
            print(out_file, runtime)
            if timing:
                f.write(f"{out_file},{runtime}\n")

def can_load_docked_file(cfg, program, file_prefix, split, item):
    i, row = item
    if program == "gnina" and not PREP_FOR_GNINA:
        docked_file = f"{file_prefix}_{split}/{i}.sdf"
    else:
        docked_file = f"{file_prefix}_{split}/{i}.pdbqt"
    full_docked_file = cfg.platform[f"crossdocked_{program}_dir"]+ "/" + docked_file
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


def finalize_crossdocked_vina(cfg, program, file_prefix):
    for split in [ "test" ]:
        screen_csv = get_crossdocked_dir(cfg) + f"/{file_prefix}_{split}.csv"
        screen_df = pd.read_csv(screen_csv)

        docked_lig_files = []
        with Pool(processes=16) as p:
            f = partial(can_load_docked_file, cfg, program, file_prefix, split)
            for res in tqdm(p.imap(f, screen_df.iterrows()), total=len(screen_df)):
                docked_lig_files.append(res)

        screen_df["docked_lig_file"] = docked_lig_files
        screen_df = screen_df.dropna().reset_index(drop=True)
        
        out_file = cfg.platform[f"crossdocked_{program}_dir"] + f"/{file_prefix}_{split}.csv"
        print(f"Saving docked df to {out_file}")
        screen_df.to_csv(out_file, index=False)

if __name__ == "__main__":

    cfg = get_config("icml")
    dock_all(cfg, "gnina", "structures", True)
    # dock_all(cfg, "vina", "structures", True)
    finalize_crossdocked_vina(cfg, "gnina", "structures")
    # finalize_crossdocked_vina(cfg, "vina", "structures")