from glob import glob
from typing import List
import pandas as pd
from rdkit import Chem
import torch
from tqdm import tqdm, trange
from common.cache import cache
from common.pose_transform import MultiPose
from common.utils import get_mol_from_file
from data_formats.transforms import lig_docked_poses, lig_embed_pose
from models.model import Model, ScoreActivityModel
from terrace.batch import Batch
from terrace.dataframe import DFRow

class DiffDock(Model):

    def __init__(self, cfg, split):
        self.cfg = cfg
        self.cache_key = "diffdock"
        self.device = "cpu"
        self.split = split
        bb_diffdock_csv = cfg.platform.diffdock_dir + f"/data/bb_struct_{split}.csv"
        df = pd.read_csv(bb_diffdock_csv)
        df["rec_file"] = df.protein_path.str.split("/").apply(lambda x: "/".join(x[-2:]))
        self.df = df

    def get_input_feats(self):
        return ["lig_docked_poses"]
    
    def get_tasks(self):
        return ["predict_lig_pose"]

    def to(self, device):
        self.device = device
        return self

    def call_single(self, x):

        lig_file = "/".join(x.lig_crystal_file.split("/")[-2:])
        rec_file = "_".join(("/".join(x.rec_file.split("/")[-2:])).split("_")[:-1]) + ".pdb"
        results = self.df.query("rec_file == @rec_file and lig_file == @lig_file").reset_index(drop=True)
        
        if len(results) == 0:
            return DFRow(lig_pose=x.lig_docked_poses)
        
        assert len(results) == 1
        
        complex_name = results.complex_name[0]

        result_folder = self.cfg.platform.diffdock_dir + f"/results/bb_struct_{self.split}/" + complex_name
        result_sdfs = glob(result_folder + "/rank*_confidence*.sdf")
        
        if len(result_sdfs) == 0:
            return DFRow(lig_pose=x.lig_docked_poses)
        
        
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

        pose = MultiPose(coord=torch.stack(coord_list).to(self.device))
        return DFRow(lig_pose=pose)
    
@cache(lambda cfg, d: d.split, disable=False)
def get_diffdock_indexes(cfg, dataset):
    model = DiffDock(cfg, dataset.split)
    indexes = set()
    for i, (x, y) in enumerate(tqdm(dataset)):
        lig_file = "/".join(x.lig_crystal_file.split("/")[-2:])
        rec_file = "_".join(("/".join(x.rec_file.split("/")[-2:])).split("_")[:-1]) + ".pdb"
        results = model.df.query("rec_file == @rec_file and lig_file == @lig_file")
        
        if len(results) == 1:
            indexes.add(i)
    return indexes
