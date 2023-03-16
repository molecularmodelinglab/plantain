import torch
from typing import Set, Type
import pandas as pd
from rdkit import Chem
from common.utils import get_mol_from_file, get_prot_from_file, get_gnina_scores_from_pdbqt
from data_formats.tasks import Task
from datasets.base_datasets import Dataset
from terrace.batch import NoStackTensor
from terrace.dataframe import DFRow

class BigBindGninaDataset(Dataset):

    def __init__(self, cfg, split, transform):
        super().__init__(cfg, transform)
        self.sna_frac = cfg.data.sna_frac
        if self.sna_frac is None:
            csv = cfg.platform.bigbind_gnina_dir + f"/activities_{split}.csv"
        else:
            csv = cfg.platform.bigbind_gnina_dir + f"/activities_sna_{self.sna_frac}_{split}.csv"
            self.activity_values = pd.read_csv(cfg.platform.bigbind_dir + f"/activities_{split}.csv")
        self.activities = pd.read_csv(csv)
        self.dir = cfg.platform.bigbind_dir
        self.gnina_dir = cfg.platform.bigbind_gnina_dir
        self.split = split

    @staticmethod
    def get_name():
        return "bigbind_gnina"

    def len_impl(self):
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

    def get_label_feats(self):
        if self.sna_frac is None:
            return { "activity"}
        return { "is_active", "activity" }

    def get_activity(self, index):
        if self.sna_frac is None:
            return self.activities[index].activity
        elif self.sna_frac == 1:
            lf = self.activities.docked_lig_file[index]
            og_index = int(lf.split("/")[-1].split(".")[0])
            if og_index % 2 == 0:
                val_idx = og_index // 2
                assert self.activities.lig_smiles[index] == self.activity_values.lig_smiles[val_idx]
                return self.activity_values.pchembl_value[val_idx]
            else:
                return torch.nan
        else:
            raise AssertionError

    def getitem_impl(self, index):

        lig_file = self.get_lig_file(index)
        rec_file = self.get_rec_file(index)

        lig = get_mol_from_file(lig_file)
        lig = Chem.RemoveHs(lig)
        rec = get_prot_from_file(rec_file)
        poc_id = self.activities.pocket[index]

        pose_scores, affinities = get_gnina_scores_from_pdbqt(lig_file)
        pose_scores = NoStackTensor(torch.tensor(pose_scores, dtype=torch.float32))
        affinities = NoStackTensor(torch.tensor(affinities, dtype=torch.float32))

        activity = self.get_activity(index)
        activity = torch.tensor(activity, dtype=torch.float32)

        x = DFRow(lig=lig, rec=rec, pocket_id=poc_id, gnina_pose_scores=pose_scores, gnina_affinities=affinities)
        if self.sna_frac is None:
            y = DFRow(activity=activity)
        else:
            y = DFRow(is_active=self.activities.active[index], activity=activity)

        return x, y