#!/usr/bin/env python

from glob import glob
import subprocess
import os

from common.cfg_utils import get_config

def preprocess_lit_pcba(cfg):
    """ Uses PyMol to select the pocket residues of the proteins in LIT-PCBA """
    dist = 5
    for target in glob(cfg.platform.lit_pcba_dir + "/*"):
        if not os.path.isdir(target): continue
        prot_files = glob(target + "/*_protein.mol2")
        pdb_ids = [ f.split("/")[-1].split("_")[0] for f in prot_files ]
        if target.endswith("IDH1"):
            # IDH1 has several pockets so we choose one
            pdb_ids = [ "5de1" ]
        out_file = target + "/process.pml"
        print(f"Writing to {out_file}")
        with open(out_file, "w") as f:
            for pdb in pdb_ids:
                f.write(f"load {pdb}_protein.mol2\n")
                f.write(f"load {pdb}_ligand.mol2\n")
                f.write(f"create {pdb}, ({pdb}_protein or {pdb}_ligand)\n")
                f.write(f"remove {pdb}_protein\n")
                f.write(f"remove {pdb}_ligand\n")
            f.write("""
                remove solvent
                remove inorganic
                """)
            to_align = pdb_ids[0]
            for pdb in pdb_ids[1:]:
                f.write(f"align {pdb}, {to_align}\n")
            for pdb in pdb_ids:
                poc = pdb + "_pocket"
                out_pdb = target + f"/{poc}.pdb"
                f.write(f"select byres (({pdb} within {dist} of organic) and not organic)\n")
                f.write(f"set_name sele, {poc}\n")
                f.write(f"save {out_pdb}, {poc}\n")
        cur_dir = os.getcwd()
        os.chdir(target)
        subprocess.run(["pymol", "-c", "process.pml"])
        os.chdir(cur_dir)

if __name__ == "__main__":
    cfg = get_config()
    preprocess_lit_pcba(cfg)