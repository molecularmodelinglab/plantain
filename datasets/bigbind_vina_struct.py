from rdkit import Chem
from rdkit.Chem import rdMolAlign
import torch
import pandas as pd
from traceback import print_exc
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from common.utils import get_mol_from_file, get_prot_from_file

from datasets.cacheable_dataset import CacheableDataset

from datasets.data_types import InteractionStructData
from datasets.graphs.interaction_graph import InteractionGraph
from datasets.graphs.mol_graph import MolGraph
from datasets.graphs.prot_graph import ProtGraph

class BigBindVinaStructDataset(CacheableDataset):

    def __init__(self, cfg, split):
        super().__init__(cfg, "bigbind_vina_struct")
        csv = cfg.platform.bigbind_vina_dir + f"/structures_{split}.csv"
        self.structures = pd.read_csv(csv)
        self.dir = cfg.platform.bigbind_dir
        self.vina_dir = cfg.platform.bigbind_vina_dir
        self.cfg = cfg
        self.split = split

    def __len__(self):
        return len(self.structures)

    def get_lig_file(self, index):
        """ returns the first lig file if use_lig is false, to ensure
        that all ligs are the same """
        if not self.cfg.data.use_lig:
            return self.dir + "/VAOX_PENSI_1_560_0/1qlt_fad_lig.sdf"
        return self.dir + "/" + self.structures.lig_file[index]

    def get_docked_lig_file(self, index):
        if not self.cfg.data.use_lig:
            return self.vina_dir + "/structures_train/0.pdbqt"
        return self.vina_dir + "/" + self.structures.docked_lig_file[index]

    def get_rec_file(self, index):
        """ same as above """
        if self.cfg.data.use_rec:
            poc_file = self.structures.ex_rec_pocket_file[index]
            rec_file = self.structures.ex_rec_file[index]
        else:
            poc_file = "VAOX_PENSI_1_560_0/1e8f_A_rec_pocket.pdb"
            rec_file = "VAOX_PENSI_1_560_0/1e8f_A_rec.pdb"
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
        docked_lig_file = self.get_docked_lig_file(index)
        rec_file = self.get_rec_file(index)
        
        try:

            lig = get_mol_from_file(lig_file)
            lig = Chem.RemoveHs(lig)

            docked_lig = get_mol_from_file(docked_lig_file)
            docked_lig = Chem.RemoveHs(docked_lig)

            rec = get_prot_from_file(rec_file)

            rec_graph = ProtGraph(self.cfg, rec)
            lig_graph = MolGraph(self.cfg, lig)
            
            rmsds = []
            inter_graphs = []
            # for now: use all conformers
            for conformer in range(docked_lig.GetNumConformers()):

                docked_lig_graph = MolGraph(self.cfg, docked_lig, conformer)
                inter_graph = InteractionGraph(self.cfg, docked_lig_graph, rec_graph)
                inter_graphs.append(inter_graph)

                rmsds.append(rdMolAlign.CalcRMS(lig, docked_lig, 0, conformer))
            
            ret = InteractionStructData(tuple(inter_graphs), lig_graph, torch.tensor(rmsds, dtype=torch.float32))
        
        except KeyboardInterrupt:
            raise
        except:
            print(f"Error proccessing item at {index=}")
            print(f"{lig_file=}")
            print(f"{rec_file=}")
            print(f"{docked_lig_file=}")
            raise

        return ret

    def get_variance(self):
        return {}

    def get_type_data(self):
        return InteractionStructData.get_type_data(self.cfg)