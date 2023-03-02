import torch
from rdkit.Chem import rdMolAlign
from typing import Set, Type
import pandas as pd
from common.utils import get_mol_from_file, get_prot_from_file, get_gnina_scores_from_pdbqt
from data_formats.tasks import Task
from data_formats.transforms import get_docked_conformers
from datasets.base_datasets import Dataset
from terrace.batch import NoStackTensor
from terrace.dataframe import DFRow

class BigBindGninaStructDataset(Dataset):

    def __init__(self, cfg, split, transform):
        super().__init__(cfg, transform)
        self.sna_frac = cfg.data.sna_frac
        csv = cfg.platform.bigbind_gnina_dir + f"/structures_{split}.csv"
        self.structures = pd.read_csv(csv)
        self.dir = cfg.platform.bigbind_dir
        self.gnina_dir = cfg.platform.bigbind_gnina_dir
        self.split = split

    @staticmethod
    def get_name():
        return "bigbind_gnina_struct"

    def len_impl(self):
        return len(self.structures)

    # smh a lot of this is brazenly copy-and-pasted from bigbind_act
    def get_lig_file(self, index):
        """ returns the first lig file if use_lig is false, to ensure
        that all ligs are the same """
        if not self.cfg.data.use_lig:
            return self.dir + "/VAOX_PENSI_1_560_0/1qlt_fad_lig.sdf"
        return self.dir + "/" + self.structures.lig_file[index]

    def get_docked_lig_file(self, index):
        if not self.cfg.data.use_lig:
            return self.gnina_dir + "/structures_train/0.pdbqt"
        return self.gnina_dir + "/" + self.structures.docked_lig_file[index]

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
        return { "pose_rmsds" }

    def getitem_impl(self, index):

        true_lig_file = self.get_lig_file(index)
        docked_lig_file = self.get_docked_lig_file(index)
        rec_file = self.get_rec_file(index)

        true_lig = get_mol_from_file(true_lig_file)
        docked_lig = get_mol_from_file(docked_lig_file)
        rec = get_prot_from_file(rec_file)
        poc_id = self.structures.pocket[index]

        pose_scores, affinities = get_gnina_scores_from_pdbqt(docked_lig_file)
        pose_scores = torch.tensor(pose_scores, dtype=torch.float32)
        affinities = torch.tensor(affinities, dtype=torch.float32)

        x = DFRow(lig=docked_lig, rec=rec, pocket_id=poc_id, gnina_pose_scores=pose_scores, gnina_affinities=affinities)
        
        rmsds = []
        for conformer in get_docked_conformers(self.cfg, docked_lig):
            rmsds.append(rdMolAlign.CalcRMS(true_lig, docked_lig, 0, conformer))
        rmsds = NoStackTensor(torch.tensor(rmsds, dtype=torch.float32))
        y = DFRow(pose_rmsds=rmsds)

        return x, y