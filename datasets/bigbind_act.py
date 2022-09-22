from rdkit import Chem
import torch
from traceback import print_exc
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning

from datasets.bigbind import BigBindDataset
from datasets.graphs.mol_graph import MolGraph, mol_graph_from_sdf
from datasets.graphs.prot_graph import ProtGraph, prot_graph_from_pdb
from datasets.data_types import ActivityData

class BigBindActDataset(BigBindDataset):

    def __init__(self, cfg, split):
        super(BigBindActDataset, self).__init__(cfg, "bigbind_act", split)

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
        
        activity = torch.tensor(self.activities.pchembl_value[index], dtype=torch.float32)

        try:
            lig_graph = mol_graph_from_sdf(self.cfg, lig_file)
            rec_graph = prot_graph_from_pdb(self.cfg, rec_file)
        except:
            print(f"Error proccessing item at {index=}")
            print(f"{lig_file=}")
            print(f"{rec_file=}")
            raise

        return ActivityData(lig_graph, rec_graph, activity)

    def get_variance(self):
        return {
            "activity": self.activities.pchembl_value.var(),
        }

    def get_type_data(self):
        return ActivityData.get_type_data(self.cfg)