import pandas as pd
from rdkit import Chem
import torch
from common.pose_transform import MultiPose
from data_formats.transforms import lig_docked_poses, lig_embed_pose
from models.model import Model
from terrace.dataframe import DFRow

class GninaComboPose(Model):

    def __init__(self, cfg, other_cache_key):
        self.cfg = cfg
        self.dir = f"outputs/pose_preds/{other_cache_key}-minimized"
        self.cache_key = f"gnina_combo_{other_cache_key}"
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
        fname = self.dir + f"/{int(x.index)}.sdf"

        coords = []
        scores = []
        for lig in Chem.SDMolSupplier(fname):
            scores.append(lig.GetPropsAsDict()["CNNscore"])
            lig = Chem.RemoveHs(lig)
            assert lig.GetNumAtoms() == x.lig.GetNumAtoms()
            order = lig.GetSubstructMatch(x.lig)
            lig = Chem.RenumberAtoms(lig, list(order))
            pose = lig_embed_pose(self.cfg, DFRow(lig=lig))
            coords.append(pose.coord)

        indexes = torch.argsort(-torch.tensor(scores))
        scores = torch.tensor(scores)[indexes]
        coords = torch.stack(coords)[indexes]
        pose = MultiPose(coord=coords.to(self.device))
        return DFRow(lig_pose=pose,
                     pose_scores=scores)