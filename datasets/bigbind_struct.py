import torch
from typing import Set, Type
import pandas as pd
from common.utils import get_mol_from_file, get_prot_from_file
from data_formats.base_formats import Activity, IsActive, LigAndRec, Pose
from data_formats.tasks import Task
from datasets.base_datasets import Dataset

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

    def get_label_classes(self) -> Set[Type[Task]]:
        return { Pose }

    def getitem_impl(self, index):

        lig_file = self.get_lig_file(index)
        rec_file = self.get_rec_file(index)

        lig = get_mol_from_file(lig_file)
        rec = get_prot_from_file(rec_file)
        poc_id = self.structures.pocket[index]

        x = LigAndRec(lig, rec, poc_id)
        
        lig_coords = []
        conformer = lig.GetConformer(0)
        for atom in lig.GetAtoms():
            point = conformer.GetAtomPosition(atom.GetIdx())
            coord = [ point.x, point.y, point.z ]
            lig_coords.append(coord)
        
        y = Pose(torch.tensor(lig_coords, dtype=torch.float32))

        return x, y