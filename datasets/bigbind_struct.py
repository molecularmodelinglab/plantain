from rdkit import Chem
import torch
from traceback import print_exc
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning

from datasets.bigbind import BigBindDataset
from datasets.graphs.mol_graph import MolGraph, mol_graph_from_sdf
from datasets.graphs.prot_graph import ProtGraph, prot_graph_from_pdb
from datasets.data_types import StructData

class BigBindStructDataset(BigBindDataset):

    def __init__(self, cfg, split):
        super(BigBindStructDataset, self).__init__(cfg, "bigbind_struct", split)

    def __len__(self):
        return len(self.structures)

    def get_lig_file(self, index):
        """ returns the first lig file if use_lig is false, to ensure
        that all ligs are the same """
        if not self.cfg.data.use_lig:
            return self.dir + "/ELNE_HUMAN_30_247_0/3q77_2hy_lig.sdf"
        return self.dir + "/" + self.structures.lig_file[index]

    def get_rec_file(self, index):
        """ same as above """
        if self.cfg.data.use_rec:
            poc_file = self.structures.ex_rec_pocket_file[index]
            rec_file = self.structures.ex_rec_file[index]
        else:
            poc_file = "ELNE_HUMAN_30_247_0/3q77_A_rec_pocket.pdb"
            rec_file = "ELNE_HUMAN_30_247_0/3q77_A_rec.pdb"
        if self.cfg.data.rec_graph.only_pocket:
            return self.dir + "/" + poc_file
        else:
            return self.dir + "/" + rec_file

    def get_cache_key(self, index):

        lig_file = self.get_lig_file(index).split("/")[-1]
        rec_file = self.get_rec_file(index).split("/")[-1]

        return lig_file + "_" + rec_file

    def get_item_pre_cache(self, index):
        
        lig_file = self.get_lig_file(index)
        rec_file = self.get_rec_file(index)

        print(lig_file, rec_file)
        
        try:
            lig_graph = mol_graph_from_sdf(self.cfg, lig_file)
            rec_graph = prot_graph_from_pdb(self.cfg, rec_file)
        except:
            print(f"Error proccessing item at {index=}")
            print(f"{lig_file=}")
            print(f"{rec_file=}")
            raise

        # center everything around lig centroid

        lig_centroid = lig_graph.ndata.coord.mean(0)
        lig_graph.ndata.coord -= lig_centroid
        rec_graph.ndata.coord -= lig_centroid

        return StructData(lig_graph, rec_graph)

    def get_variance(self):
        return {}

    def get_type_data(self):
        return StructData.get_type_data(self.cfg)