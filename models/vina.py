import pandas as pd
from rdkit import Chem
from common.pose_transform import MultiPose
from common.utils import get_mol_from_file
from data_formats.transforms import lig_docked_poses
from models.model import Model, ScoreActivityModel
from terrace.batch import Batch
from terrace.dataframe import DFRow

class VinaPose(Model):

    def __init__(self, cfg):
        self.cfg = cfg
        dfs = []
        for split in ["val", "test"]:
            dfs.append(pd.read_csv(cfg.platform.crossdocked_vina_dir + f"/structures_{split}.csv"))
        self.df = pd.concat(dfs)
        self.cache_key = "vina"
        self.device = "cpu"

    def get_input_feats(self):
        # this is a hack. "lig_docked_poses" will default to just using the UFF
        # pose on this dataset repeated a bunch of times. We can use that if Vina
        # didn't give us a valid pose for a particular datapoint
        return ["lig_docked_poses"]

    def get_tasks(self):
        return ["predict_lig_pose"]

    def to(self, device):
        self.device = device
        return self

    def call_single(self, x):
        lig_file = "/".join(x.lig_crystal_file.split("/")[-2:])
        rec_file = "/".join(x.rec_file.split("/")[-2:])
        results = self.df.query("crossdock_rec_pocket_file == @rec_file and lig_crystal_file == @lig_file")
        if len(results) == 0:
            return DFRow(lig_pose=x.lig_docked_poses)
        assert len(results) == 1
        
        docked_file = self.cfg.platform.crossdocked_vina_dir + "/" + next(iter(results.docked_lig_file))
        lig = get_mol_from_file(docked_file)
        lig = Chem.RemoveHs(lig)
        order = lig.GetSubstructMatch(x.lig)
        lig = Chem.RenumberAtoms(lig, list(order))
        assert lig.GetNumAtoms() == x.lig.GetNumAtoms()
        
        pose = lig_docked_poses(self.cfg, DFRow(lig=lig))
        pose = MultiPose(coord=pose.coord.to(self.device))
        return DFRow(lig_pose=pose)