import torch
import os

from datasets.cacheable_dataset import CacheableDataset
from datasets.data_types import IsActiveData
from datasets.graphs.mol_graph import MolGraph, mol_graph_from_sdf
from datasets.graphs.prot_graph import ProtGraph, prot_graph_from_pdb

class PDBBindDataset(CacheableDataset):
    """ Only supports 2016 core set for speed testing """

    def __init__(self, cfg):
        super().__init__(cfg, "pdbbind")
        self.cfg = cfg
        self.dir = cfg.platform.pdbbind_dir
        self.pdb_ids = []
        with open("data/pdb_2016_core_set_ids.txt") as f:
            for line in f.readlines():
                pdb_id = line.split("/")[0].upper()
                if not os.path.exists(self.dir + "/" + pdb_id):
                    continue
                self.pdb_ids.append(pdb_id)

    def __len__(self):
        return len(self.pdb_ids)

    def get_cache_key(self, index):
        return self.pdb_ids[index]

    def get_item_pre_cache(self, index):
        pdb_id = self.pdb_ids[index]
        rec_file = self.dir + f"/{pdb_id}/pocket.pdb" # {pdb_id}_PRO.pdb"
        lig_file = self.dir + f"/{pdb_id}/{pdb_id}_LIG.sdf"

        lig_graph = mol_graph_from_sdf(self.cfg, lig_file)
        rec_graph = prot_graph_from_pdb(self.cfg, rec_file)
        is_active = torch.tensor(False, dtype=bool)

        return IsActiveData(lig_graph, rec_graph, is_active)