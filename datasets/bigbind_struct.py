import os
import torch
from tqdm import trange
from typing import Set, Type
import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools
from common.cache import cache
from common.pose_transform import Pose
from data_formats.graphs.mol_graph import get_mol_coords
from data_formats.transforms import lig_crystal_pose
from common.utils import get_mol_from_file, get_prot_from_file
from data_formats.tasks import Task
from datasets.base_datasets import Dataset
from terrace.dataframe import DFRow

@cache(lambda cfg, x: x)
def get_refined_mask(cfg, csv):
    """ Returns a mask for the 'refined' version of the dataset. That is,
    all the datapoints where the ligand has a QED score of greater than 0.5
    and doesn't contain phosphorus """
    df = pd.read_csv(csv)
    PandasTools.AddMoleculeColumnToFrame(df, "lig_smiles", "lig", includeFingerprints=False)
    qed_score = df.lig.apply(Chem.Descriptors.qed)
    mask = (qed_score > 0.5) & (~df.lig_smiles.str.contains("P"))
    return mask

class BigBindStructDataset(Dataset):

    def get_bad_indexes(self):
        # some jank to use the UFF files from the V2 dataset with the V1
        bad_indexes = []
        for index in trange(len(self.structures)):
            uff_file = self.get_lig_uff_file(index)
            if not os.path.exists(uff_file):
                bad_indexes.append(index)
                continue

            # there's some weird stuff going on here...
            # todo: figure out why there's all these charge issues
            # and only on the test set! 
            lig = get_mol_from_file(uff_file)
            lig = Chem.RemoveHs(lig)

            uff_smiles = Chem.MolToSmiles(lig, False)
            reg_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(self.structures.lig_smiles[index]), False)

            if uff_smiles != reg_smiles:
                bad_indexes.append(index)

        return bad_indexes

    def __init__(self, cfg, split, transform):
        super().__init__(cfg, transform)
        csv = cfg.platform.bigbind_dir + f"/structures_{split}.csv"
        self.structures = pd.read_csv(csv)
        self.refined_mask = get_refined_mask(cfg, csv)
        
        # if split == "val":
        #     bb_diffdock_csv = cfg.platform.diffdock_dir + "/data/bb_struct_val.csv"
        #     self.diffdock_indexes = set(pd.read_csv(bb_diffdock_csv)["Unnamed: 0"])
        # else:
        #     self.diffdock_indexes = set()

        max_residues = self.cfg.data.get("max_rec_residues", None)
        if max_residues is not None:
            self.structures = self.structures.query("num_pocket_residues <= @max_residues").reset_index(drop=True)

        self.dir = cfg.platform.bigbind_dir
        self.v2_dir = cfg.platform.bigbind_struct_v2_dir
        self.split = split

        bad_indexes = self.get_bad_indexes()

        self.structures = self.structures.drop(bad_indexes).reset_index(drop=True)

    @staticmethod
    def get_name():
        return "bigbind_struct"

    def len_impl(self):
        return len(self.structures)

    def get_lig_crystal_file(self, index):
        """ returns the first lig file if use_lig is false, to ensure
        that all ligs are the same """
        if not self.cfg.data.use_lig:
            return self.dir + "/ELNE_HUMAN_30_247_0/3q77_2hy_lig.sdf"
        return self.dir + "/" + self.structures.lig_file[index]

    def get_lig_uff_file(self, index):
        """ returns the first lig file if use_lig is false, to ensure
        that all ligs are the same """
        if not self.cfg.data.use_lig:
            lf = "/ELNE_HUMAN_30_247_0/3q77_2hy_lig.sdf"
        else:
            lf = self.structures.lig_file[index]
        return self.v2_dir + "/" + lf.split(".")[0] + "_uff.sdf"

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
        return ["lig_crystal_pose"]

    def getitem_impl(self, index):

        rec_file = self.get_rec_file(index)
        lig_crystal_file = self.get_lig_crystal_file(index)

        # print("Getting", index, lig_file, rec_file)

        lig_crystal = get_mol_from_file(lig_crystal_file)
        lig_crystal = Chem.RemoveHs(lig_crystal)
        lig_crystal_pose = Pose(get_mol_coords(lig_crystal, 0))

        lig = get_mol_from_file(self.get_lig_uff_file(index))
        lig = Chem.RemoveHs(lig)

        order = lig.GetSubstructMatch(lig_crystal)
        lig = Chem.RenumberAtoms(lig, list(order))
        lig_embed_crystal_pose = lig_crystal_pose

        rec = get_prot_from_file(rec_file)
        poc_id = self.structures.pocket[index]

        if self.cfg.get("debug_crystal_pose_cheat", False):
            lig = lig_crystal

        refined = torch.tensor(self.refined_mask[index], dtype=torch.bool)
        # in_diffdock = torch.tensor(index in self.diffdock_indexes, dtype=torch.bool)

        x = DFRow(lig=lig,
                  rec=rec,
                  index=torch.tensor(index, dtype=torch.long),
                  pocket_id=poc_id,
                  refined=refined,
                  # in_diffdock=in_diffdock,
                  rec_file=rec_file,
                  lig_crystal_file=lig_crystal_file)
        y = DFRow(lig_crystal_pose=lig_crystal_pose, lig_embed_crystal_pose=lig_embed_crystal_pose)

        return x, y