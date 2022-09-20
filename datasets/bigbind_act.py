import warnings
from rdkit import Chem
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning

from datasets.bigbind import BigBindDataset
from datasets.graphs.mol_graph import MolGraph
from datasets.graphs.prot_graph import ProtGraph

class BigBindActDataset(BigBindDataset):

    def __init__(self, cfg, split):
        super(BigBindActDataset, self).__init__(cfg, "bigbind_act", split)

    def __len__(self):
        return len(self.activities)

    def get_cache_key(self, index):

        lig_file = self.activities.lig_file[index].split("/")[-1]
        rec_file = self.activities.ex_rec_file[index].split("/")[-1]

        return lig_file + "_" + rec_file

    def get_item_pre_cache(self, index):
        lig_file = self.dir + "/" + self.activities.lig_file[index]
        rec_file = self.dir + "/" + self.activities.ex_rec_pocket_file[index]
        activity = self.activities.pchembl_value[index]
        
        lig = next(Chem.SDMolSupplier(lig_file, sanitize=True))

        parser = PDBParser()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=PDBConstructionWarning)
            structure = parser.get_structure('random_id', rec_file)
            rec = structure[0]

        lig_graph = MolGraph(self.cfg, lig)
        rec_graph = ProtGraph(self.cfg, rec)

        return lig_graph, rec_graph, activity