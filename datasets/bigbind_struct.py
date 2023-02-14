import torch
from typing import Set, Type
import pandas as pd
from rdkit import Chem
from data_formats.transforms import lig_crystal_pose
from common.utils import get_mol_from_file, get_prot_from_file
from data_formats.tasks import Task
from datasets.base_datasets import Dataset
from terrace.dataframe import DFRow

class BigBindStructDataset(Dataset):

    def __init__(self, cfg, split, transform):
        super().__init__(cfg, transform)
        csv = cfg.platform.bigbind_dir + f"/structures_{split}.csv"
        self.structures = pd.read_csv(csv)
        self.dir = cfg.platform.bigbind_dir
        self.split = split

    @staticmethod
    def get_name():
        return "bigbind_struct"

    def len_impl(self):
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

    def get_label_feats(self) -> Set[Type[Task]]:
        return ["lig_crystal_pose"]

    def getitem_impl(self, index):

        lig_file = self.get_lig_file(index)
        rec_file = self.get_rec_file(index)

        # print("Getting", index, lig_file, rec_file)

        lig = get_mol_from_file(lig_file)
        lig = Chem.RemoveHs(lig)

        rec = get_prot_from_file(rec_file)
        poc_id = self.structures.pocket[index]

        x = DFRow(lig=lig, rec=rec, pocket_id=poc_id)
        
        y = DFRow(lig_crystal_pose=lig_crystal_pose(self.cfg, x))

        return x, y