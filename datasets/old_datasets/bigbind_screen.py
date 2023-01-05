import os
import pandas as pd
from glob import glob
from rdkit import Chem
import torch
from traceback import print_exc
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning

from datasets.cacheable_dataset import CacheableDataset
from datasets.data_types import IsActiveIndexData

class BigBindScreenDataset(CacheableDataset):

    @staticmethod
    def get_all_targets(cfg, split):
        csv_files = glob(cfg.platform.bigbind_dir + f"/{split}_screens/*.csv")
        return [ f.split("/")[-1].split(".")[0] for f in csv_files ]

    def __init__(self, cfg, target, split):
        name = "bigbind_screen_" + target
        super().__init__(cfg, name)
        self.split = split
        self.target = target
        self.cfg = cfg
        self.dir = cfg.platform.bigbind_dir
        csv_file = self.dir + f"/{split}_screens/{target}.csv"
        self.activities = pd.read_csv(csv_file)
        self.activities = self.activities[self.activities.lig_smiles.str.len() > 5].reset_index(drop=True)
        max_pocket_size=42
        self.activities = self.activities.query("num_pocket_residues >= 5 and pocket_size_x < @max_pocket_size and pocket_size_y < @max_pocket_size and pocket_size_z < @max_pocket_size").reset_index(drop=True)

    def __len__(self):
        return len(self.activities)

    def get_lig_file(self, index):
        """ returns the first lig file if use_lig is false, to ensure
        that all ligs are the same """
        if not self.cfg.data.use_lig:
            index = 0
        return self.dir + "/" + self.activities.lig_file[index]

    def get_rec_file(self, index):
        """ same as above """
        if not self.cfg.data.use_rec:
            index = 0
        if self.cfg.data.rec_graph.only_pocket:
            return self.dir + "/" + self.activities.ex_rec_pocket_file[index]
        else:
            return self.dir + "/" + self.activities.ex_rec_file[index]

    def get_cache_key(self, index):

        lig_file = self.get_lig_file(index).split("/")[-1]
        rec_file = self.get_rec_file(index).split("/")[-1]

        return lig_file + "_" + rec_file

    def get_item_pre_cache(self, index):
        
        lig_file = self.get_lig_file(index)
        rec_file = self.get_rec_file(index)

        try:
            lig_graph = mol_graph_from_sdf(self.cfg, lig_file)
            rec_graph = prot_graph_from_pdb(self.cfg, rec_file)
        except:
            print(f"Error proccessing item at {index=}")
            print(f"{lig_file=}")
            print(f"{rec_file=}")
            raise
        
        is_active = torch.tensor(self.activities.active[index], dtype=bool)
        return IsActiveIndexData(lig_graph, rec_graph, is_active, index)

    def get_all_yt(self):
        return torch.tensor(self.activities.active, dtype=bool) 

    def get_variance(self):
        return {}

    def get_type_data(self):
        return IsActiveIndexData.get_type_data(self.cfg)