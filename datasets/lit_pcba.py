import os
from rdkit import Chem
import torch
from glob import glob
from datasets.cacheable_dataset import CacheableDataset
from datasets.graphs.mol_graph import MolGraph
from datasets.graphs.prot_graph import ProtGraph, prot_graph_from_pdb
from datasets.data_types import IsActiveIndexData

class LitPcbaDataset(CacheableDataset):

    @staticmethod
    def get_all_targets(cfg):
        folders = glob(cfg.platform.lit_pcba_dir + "/*")
        return [ folder.split("/")[-1] for folder in folders if os.path.isdir(folder) ]

    def __init__(self, cfg, target):
        name = "lit_pcba_" + target
        super().__init__(cfg, name)
        self.cfg = cfg
        self.target = target
        self.dir = cfg.platform.lit_pcba_dir + "/" + target
        with open(self.dir + "/actives.smi", "r") as f:
            actives = f.readlines()
        with open(self.dir + "/inactives.smi", "r") as f:
            inactives = f.readlines()

        self.items = []
        are_active = [ True ]*len(actives) + [ False ]*len(inactives)
        for i, (line, active) in enumerate(zip(actives + inactives, are_active)):
            smi, pcba_idx = line.split()
            pcba_idx = int(pcba_idx)
            self.items.append((smi, active, pcba_idx))

        poc_file = glob(self.dir + "/*_pocket.pdb")[0]
        self.prot_graph = prot_graph_from_pdb(cfg, poc_file)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        smiles, is_active, pcba_idx = self.items[index]
        mol = Chem.MolFromSmiles(smiles)
        lig_graph = MolGraph(self.cfg, mol, use_3d=False)
        is_active = torch.tensor(is_active, dtype=bool)

        return IsActiveIndexData(lig_graph, self.prot_graph, is_active, pcba_idx)

    def get_variance(self):
        return {}

    def get_type_data(self):
        return IsActiveIndexData.get_type_data(self.cfg)

    def get_all_yt(self):
        return torch.tensor([item[1] for item in self.items], dtype=bool)        

