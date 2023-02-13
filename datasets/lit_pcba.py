import os
from rdkit import Chem
import torch
from glob import glob
from datasets.base_datasets import Dataset
from common.utils import get_mol_from_file, get_prot_from_file

class LitPcbaDataset(Dataset):

    @staticmethod
    def get_all_targets(cfg):
        folders = glob(cfg.platform.lit_pcba_dir + "/*")
        return [ folder.split("/")[-1] for folder in folders if os.path.isdir(folder) ]

    @staticmethod
    def get_name():
        return "lit_pcba"

    def get_label_classes(self):
        return { IsActive }

    def __init__(self, cfg, target, transform):
        super().__init__(cfg, transform)
        name = "lit_pcba_" + target
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

        assert self.cfg.data.rec_graph.only_pocket
        poc_file = glob(self.dir + "/*_pocket.pdb")[0]
        self.rec = get_prot_from_file(poc_file)

        dense = False
        self.gnina_scores = {}
        score_file = "./prior_work/lit-pcba_dense-CNNaffinity-mean-then-max.summary" if dense else "./prior_work/newdefault_CNNaffinity-max.summary"
        with open(score_file) as f:
            for line in f.readlines():
                _, score, target, idx, _ = line.split()
                if target != self.target: continue
                self.gnina_scores[int(idx)] = float(score)


    def len_impl(self):
        return len(self.items)

    def getitem_impl(self, index):

        smiles, is_active, pcba_idx = self.items[index]
        lig = Chem.MolFromSmiles(smiles)
        is_active = torch.tensor(is_active, dtype=bool)

        pose_scores = [ torch.tensor(0.0, dtype=torch.float32) ]
        if pcba_idx in self.gnina_scores:
            score = self.gnina_scores[pcba_idx]
        else:
            score = 5.0
        affinities = [ torch.tensor(score, dtype=torch.float32) ]

        x = LigAndRecGnina(lig, self.rec, self.target, pose_scores, affinities)
        y = IsActive(is_active)

        return x, y