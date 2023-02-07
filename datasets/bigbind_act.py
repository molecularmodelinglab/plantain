from glob import glob
import random
import torch
from typing import Set, Type
import pandas as pd
from common.utils import get_mol_from_file, get_prot_from_file
from data_formats.base_formats import Activity, Data, Input, IsActive, LigAndRec
from data_formats.tasks import Task
from datasets.base_datasets import Dataset

class ProbisSim(Input):
    train_probis_similarity: float

class BigBindActDataset(Dataset):

    def __init__(self, cfg, split, transform):
        super().__init__(cfg, transform)
        self.sna_frac = cfg.data.sna_frac
        if self.sna_frac is None:
            csv = cfg.platform.bigbind_dir + f"/activities_{split}.csv"
        else:
            csv = cfg.platform.bigbind_dir + f"/activities_sna_{self.sna_frac}_{split}.csv"
        self.activities = pd.read_csv(csv)
        self.dir = cfg.platform.bigbind_dir
        self.split = split

    @staticmethod
    def get_name():
        return "bigbind_act"

    def len_impl(self):
        return len(self.activities)

    def get_lig_file(self, index):
        """ returns the first lig file if use_lig is false, to ensure
        that all ligs are the same """
        if not self.cfg.data.use_lig:
            return self.dir + "/" + "chembl_structures/mol_4.sdf"

        # todo: generalize to more sna fractions
        if self.sna_frac is not None and self.split == "train":
            assert self.sna_frac == 1
            if index % 2 == 1:
                cur_cluster = self.activities.sna_cluster[index]
                assert self.activities.active[index] == False
                # instead of using the SNA ligand the df gives us, generate one online
                while True:
                    # generate odd index
                    index = int(random.random()*len(self.activities)/2)*2
                    new_cluster = self.activities.sna_cluster[index]
                    if new_cluster != cur_cluster:
                        break

        return self.dir + "/" + self.activities.lig_file[index]

    def get_rec_file(self, index):
        """ same as above """
        if self.cfg.data.use_rec:
            if False: #self.split == "train":
                poc_folder = self.dir + "/" + self.activities.pocket[index]
                all_poc_files = glob(f"{poc_folder}/*_pocket.pdb")
                all_rec_files = glob(f"{poc_folder}/*_rec.pdb")
                poc_file = random.choice(all_poc_files)
                rec_file = random.choice(all_rec_files)
            else:
                poc_file = self.dir + "/" + self.activities.ex_rec_pocket_file[index]
                rec_file = self.dir + "/" + self.activities.ex_rec_file[index]
        else:
            poc_file = self.dir + "/ELNE_HUMAN_30_247_0/3q77_A_rec_pocket.pdb"
            rec_file = self.dir + "/ELNE_HUMAN_30_247_0/3q77_A_rec.pdb"
        if self.cfg.data.rec_graph.only_pocket:
            return poc_file
        else:
            return rec_file

    def get_label_classes(self) -> Set[Type[Task]]:
        if self.sna_frac is None:
            return { Activity }
        return { IsActive }

    def getitem_impl(self, index):

        lig_file = self.get_lig_file(index)
        rec_file = self.get_rec_file(index)

        lig = get_mol_from_file(lig_file)
        rec = get_prot_from_file(rec_file)
        poc_id = self.activities.pocket[index]

        x = LigAndRec(lig, rec, poc_id)
        if self.sna_frac is None:
            y = Activity(torch.tensor(self.activities.pchembl_value[index], dtype=torch.float32))
        else:
            y = IsActive(self.activities.active[index])

        if self.split == "val":
            x_probis = ProbisSim(self.activities.train_probis_similarity[index])
            x = Data.merge([x, x_probis])

        return x, y
