import torch
from typing import Set, Type
import pandas as pd
from common.utils import get_mol_from_file, get_prot_from_file, get_gnina_scores_from_pdbqt
from data_formats.base_formats import Activity, IsActive, LigAndRec
from data_formats.tasks import Task
from datasets.base_datasets import Dataset

class LigAndRecGnina(LigAndRec):

    pose_scores: torch.Tensor
    affinities: torch.Tensor

    @staticmethod
    def collate_pose_scores(x):
        return x

    @staticmethod
    def collate_affinities(x):
        return x

class BigBindGninaDataset(Dataset):

    def __init__(self, cfg, split, transform):
        super().__init__(cfg, transform)
        self.sna_frac = cfg.data.sna_frac
        if self.sna_frac is None:
            csv = cfg.platform.bigbind_gnina_dir + f"/activities_{split}.csv"
        else:
            csv = cfg.platform.bigbind_gnina_dir + f"/activities_sna_{self.sna_frac}_{split}.csv"
        self.activities = pd.read_csv(csv)
        self.dir = cfg.platform.bigbind_dir
        self.gnina_dir = cfg.platform.bigbind_gnina_dir
        self.split = split

    @staticmethod
    def get_name():
        return "bigbind_gnina"

    def __len__(self):
        return len(self.activities)

    # smh a lot of this is brazenly copy-and-pasted from bigbind_act
    def get_lig_file(self, index):
        """ returns the first lig file if use_lig is false, to ensure
        that all ligs are the same """
        if not self.cfg.data.use_lig:
            return self.gnina_dir + "/activities_sna_1_train/0.pdbqt"
        return self.gnina_dir + "/" + self.activities.docked_lig_file[index]

    def get_rec_file(self, index):
        """ same as above """
        if self.cfg.data.use_rec:
            poc_file = self.activities.ex_rec_pocket_file[index]
            rec_file = self.activities.ex_rec_file[index]
        else:
            poc_file = "ELNE_HUMAN_30_247_0/3q77_A_rec_pocket.pdb"
            rec_file = "ELNE_HUMAN_30_247_0/3q77_A_rec.pdb"
        if self.cfg.data.rec_graph.only_pocket:
            return self.dir + "/" + poc_file
        else:
            return self.dir + "/" + rec_file

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

        pose_scores, affinities = get_gnina_scores_from_pdbqt(lig_file)
        pose_scores = torch.tensor(pose_scores, dtype=torch.float32)
        affinities = torch.tensor(affinities, dtype=torch.float32)

        x = LigAndRecGnina(lig, rec, poc_id, pose_scores, affinities)
        if self.sna_frac is None:
            y = Activity(torch.tensor(self.activities.pchembl_value[index], dtype=torch.float32))
        else:
            y = IsActive(self.activities.active[index])

        return x, y