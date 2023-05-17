import pandas as pd
from rdkit import Chem
import torch
from common.pose_transform import MultiPose
from common.utils import get_gnina_scores_from_pdbqt, get_mol_from_file
from data_formats.transforms import lig_docked_poses
from models.model import Model, ScoreActivityModel
from terrace.batch import Batch
from terrace.dataframe import DFRow

class Gnina(ScoreActivityModel):

    def __init__(self, conf_id: int):
        self.conf_id = conf_id

    def call_single(self, x: DFRow):
        return DFRow(score=x.gnina_affinities[self.conf_id], pose_scores=x.gnina_pose_scores[self.conf_id])

    def get_input_feats(self):
        return ["gnina_affinities", "gnina_pose_scores"]

    @staticmethod
    def get_name() -> str:
        return "gnina"

    def get_tasks(self):
        return [ "score_activity_regr", "score_activity_class", "score_pose" ]

class GninaPose(Model):

    def __init__(self, cfg):
        self.cfg = cfg
        dfs = []
        for split in ["train", "val", "test"]:
            dfs.append(pd.read_csv(cfg.platform.bigbind_gnina_dir + f"/structures_{split}.csv"))
        self.df = pd.concat(dfs)
        self.cache_key = "gnina"
        self.device = "cpu"

    def get_input_feats(self):
        # this is a hack. "lig_docked_poses" will default to just using the UFF
        # pose on this dataset repeated a bunch of times. We can use that if GNINA
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
        results = self.df.query("ex_rec_pocket_file == @rec_file and lig_file == @lig_file")
        
        all_pose_scores = torch.zeros((self.cfg.data.num_poses,), device=self.device)
        
        if len(results) == 0:
            return DFRow(lig_pose=x.lig_docked_poses,
                         pose_scores=all_pose_scores)
        assert len(results) == 1
        
        docked_file = self.cfg.platform.bigbind_gnina_dir + "/" + next(iter(results.docked_lig_file))
        pose_scores, affinities = get_gnina_scores_from_pdbqt(docked_file)

        for i, score in enumerate(pose_scores):
            if i >= len(all_pose_scores): break
            all_pose_scores[i] = score

        lig = get_mol_from_file(docked_file)
        lig = Chem.RemoveHs(lig)
        assert lig.GetNumAtoms() == x.lig.GetNumAtoms()

        order = lig.GetSubstructMatch(x.lig)

        try:
            lig = Chem.RenumberAtoms(lig, list(order))
        except ValueError:
            print("Getting the charge issue again")
            return DFRow(lig_pose=x.lig_docked_poses,
                         pose_scores=all_pose_scores)

        
        pose = lig_docked_poses(self.cfg, DFRow(lig=lig))
        pose = MultiPose(coord=pose.coord.to(self.device))
        return DFRow(lig_pose=pose,
                     pose_scores=all_pose_scores)