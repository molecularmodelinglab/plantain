from collections import defaultdict
import pandas as pd
from tqdm import tqdm
from common.cfg_utils import get_config
from datasets.bigbind_struct import BigBindStructDataset
from datasets.bigbind_struct_v2 import BigBindStructV2Dataset


def main(cfg, split, version):
    if version == 2:
        dataset = BigBindStructV2Dataset(cfg, split, [])
    else: 
        dataset = BigBindStructDataset(cfg, split, [])    
    val_struct_df = dataset.structures
    dd_train_set = [ line.strip() for line in open(cfg.platform.diffdock_dir + "/data/splits/timesplit_no_lig_overlap_train").readlines() ]

    sifts_file = "/home/boris/Data/SIFTS/pdb_chain_uniprot.csv"
    sifts_df = pd.read_csv(sifts_file, comment='#')
    
    chain2uniprot = {}
    pdb_to_uniprots = defaultdict(set)
    for i, row in tqdm(sifts_df.iterrows(), total=len(sifts_df)):
        chain2uniprot[(row["PDB"], row["CHAIN"])] = row["SP_PRIMARY"]
        pdb_to_uniprots[row["PDB"]].add(row["SP_PRIMARY"])

    dd_train_uniprots = set()
    for pdb in dd_train_set:
        dd_train_uniprots = dd_train_uniprots.union(pdb_to_uniprots[pdb])

    if version == 2:
        rf_col =  val_struct_df.crossdock_rec_file
    else:
        rf_col = val_struct_df.ex_rec_file

    val_chains = rf_col.str.split("_").apply(lambda x: x[-2])
    val_pdbs = rf_col.str.split("_").apply(lambda x: x[-3].split("/")[-1])

    bb_val_uniprots = []
    for pdb, chain in zip(val_pdbs, val_chains):
        try:
            bb_val_uniprots.append(chain2uniprot[(pdb, chain)])
        except KeyError:
            bb_val_uniprots.append("???")
    val_struct_df["uniprot"] = bb_val_uniprots

    valid_mask = val_struct_df.uniprot.apply(lambda u: (u not in dd_train_uniprots) and u != "???")
    valid_df = val_struct_df.loc[valid_mask]# .reset_index(drop=True)
    print(len(valid_df), valid_df.pocket.unique().shape)

    if version == 2:
        valid_rf_col =  valid_df.crossdock_rec_file
    else:
        valid_rf_col = valid_df.ex_rec_file

    rec_file = cfg.platform.bigbind_dir + "/" + valid_rf_col
    smiles = valid_df.lig_smiles
    complex_names = [ f"complex_{i}" for i in range(len(smiles))]

    if version == 2:
        lig_file =  valid_df.lig_crystal_file
    else:
        lig_file =  valid_df.lig_file

    out_df = pd.DataFrame({ "complex_name": complex_names,
                        "protein_path": rec_file, 
                        "ligand_description": smiles,
                        "lig_file": lig_file,
                        "protein_sequence": ["" for i in range(len(smiles))] })
    
    out_file = cfg.platform.diffdock_dir + f"/data/bb_struct_{split}.csv"
    print(f"Saving to {out_file}")
    out_df.to_csv(out_file, index=True)

if __name__ == "__main__":
    cfg = get_config("diffusion_v2")
    main(cfg, "test", 2)