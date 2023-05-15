from glob import glob
from typing import List
import pandas as pd
from rdkit import Chem
import torch
from common.pose_transform import MultiPose
from common.utils import get_mol_from_file
from data_formats.transforms import lig_docked_poses, lig_embed_pose
from models.model import Model, ScoreActivityModel
from terrace.batch import Batch
from terrace.dataframe import DFRow

class DiffDock(Model):

    def __init__(self, cfg):
        self.cfg = cfg
        self.cache_key = "diffdock"
        self.device = "cpu"
        bb_diffdock_csv = cfg.platform.diffdock_dir + "/data/bb_struct_val.csv"
        df = pd.read_csv(bb_diffdock_csv)
        self.df = df.set_index("Unnamed: 0")

    def get_input_feats(self):
        return ["lig_docked_poses"]
    
    def get_tasks(self):
        return ["predict_lig_pose"]

    def to(self, device):
        self.device = device
        return self
    
    def call_single(self, x):
        index = int(x.index)
        try:
            complex_name = self.df.complex_name[index]
        except KeyError:
            return DFRow(lig_pose=x.lig_docked_poses)

        result_folder = self.cfg.platform.diffdock_dir + "/results/bb_struct_val/" + complex_name
        result_sdfs = glob(result_folder + "/rank*_confidence*.sdf")
        result_sdfs = sorted(result_sdfs, key=lambda f: int(f.split("_")[-2].split("rank")[-1]))

        # print(complex_name, result_folder, len(result_sdfs))
        assert len(result_sdfs) == 40

        coord_list = []
        for sdf in result_sdfs:
            lig = get_mol_from_file(sdf)
            lig = Chem.RemoveHs(lig)
            order = lig.GetSubstructMatch(x.lig)
            lig = Chem.RenumberAtoms(lig, list(order))
            assert lig.GetNumAtoms() == x.lig.GetNumAtoms()

            coord_list.append(lig_embed_pose(self.cfg, DFRow(lig=lig)).coord)

        pose = MultiPose(coord=torch.stack(coord_list))
        return DFRow(lig_pose=pose)