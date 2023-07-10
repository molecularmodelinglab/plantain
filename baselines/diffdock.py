from collections import defaultdict
import os
import subprocess
import pandas as pd
from tqdm import tqdm
from common.cfg_utils import get_config
from datasets.crossdocked import CrossDockedDataset


def make_diffdock_dataset(cfg, split):

    out_file = cfg.platform.diffdock_dir + f"/data/crossdocked_{split}.csv"
    if os.path.exists(out_file):
        return out_file

    dataset = CrossDockedDataset(cfg, split, [])
    val_struct_df = dataset.structures
    dd_train_set = [ line.strip() for line in open(cfg.platform.diffdock_dir + "/data/splits/timesplit_no_lig_overlap_train").readlines() ]

    sifts_file = cfg.platform.sifts_csv
    sifts_df = pd.read_csv(sifts_file, comment='#')
    
    chain2uniprot = {}
    pdb_to_uniprots = defaultdict(set)
    for i, row in tqdm(sifts_df.iterrows(), total=len(sifts_df)):
        chain2uniprot[(row["PDB"], row["CHAIN"])] = row["SP_PRIMARY"]
        pdb_to_uniprots[row["PDB"]].add(row["SP_PRIMARY"])

    dd_train_uniprots = set()
    for pdb in dd_train_set:
        dd_train_uniprots = dd_train_uniprots.union(pdb_to_uniprots[pdb])

    rf_col =  val_struct_df.crossdock_rec_file

    val_chains = rf_col.str.split("_").apply(lambda x: x[-2])
    val_pdbs = rf_col.str.split("_").apply(lambda x: x[-3].split("/")[-1])

    bb_val_uniprots = []
    for pdb, chain in zip(val_pdbs, val_chains):
        try:
            bb_val_uniprots.append(chain2uniprot[(pdb, chain)])
        except KeyError:
            # This seems to be happening because PDBs that have been removed
            # Just ignore them
            bb_val_uniprots.append("???")
    val_struct_df["uniprot"] = bb_val_uniprots

    valid_mask = val_struct_df.uniprot.apply(lambda u: (u not in dd_train_uniprots) and u != "???")
    valid_df = val_struct_df.loc[valid_mask]# .reset_index(drop=True)
    print(len(valid_df), valid_df.pocket.unique().shape)

    valid_rf_col =  valid_df.crossdock_rec_file

    rec_file = cfg.platform.crossdocked_dir + "/" + valid_rf_col
    smiles = valid_df.lig_smiles
    complex_names = [ f"complex_{i}" for i in range(len(smiles))]

    lig_file =  valid_df.lig_crystal_file

    out_df = pd.DataFrame({ "complex_name": complex_names,
                        "protein_path": rec_file, 
                        "ligand_description": smiles,
                        "lig_file": lig_file,
                        "protein_sequence": ["" for i in range(len(smiles))] })
    
    print(f"Saving to {out_file}")
    out_df.to_csv(out_file, index=True)

    return out_file

if __name__ == "__main__":
    cfg = get_config("icml")
    split = "test"
    make_diffdock_dataset(cfg, split)
    os.chdir(cfg.platform.diffdock_dir)
    cmd = f"""
    conda run -n {cfg.platform.diffdock_env} \
    /usr/bin/time -o diffdock_timer_crossdocked_{split}.txt \
    python -m inference \
        --protein_ligand_csv data/crossdocked_{split}.csv \
        --out_dir results/crossdocked_{split} \
        --inference_steps 20 --samples_per_complex 40 \
        --batch_size 2 --actual_steps 18 --no_final_step_noise
    """
    print(cmd)
    #subprocess.run(cmd, shell=True)