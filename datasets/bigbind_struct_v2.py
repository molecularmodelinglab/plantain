from copy import deepcopy
import torch
from typing import Set, Type
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from common.pose_transform import MultiPose, Pose, align_poses
from common.torsion import TorsionData
from data_formats.graphs.mol_graph import get_mol_coords
from common.utils import get_mol_from_file, get_mol_from_file_no_cache, get_prot_from_file
from data_formats.tasks import Task
from datasets.base_datasets import Dataset
from terrace.dataframe import DFRow
from validation.metrics import get_rmsds

def canonicalize(mol):

    order = tuple(zip(*sorted([(j, i) for i, j in enumerate(Chem.CanonicalRankAtoms(mol, includeChirality=False))])))[1]
    mol = Chem.RenumberAtoms(mol, list(order))

    return mol

def gen_conformers(mol, numConfs=100, maxAttempts=1000, pruneRmsThresh=0.1, useExpTorsionAnglePrefs=True, useBasicKnowledge=True, enforceChirality=True):
    mol = Chem.AddHs(mol)
    ids = AllChem.EmbedMultipleConfs(mol, numConfs=numConfs, maxAttempts=maxAttempts, pruneRmsThresh=pruneRmsThresh, useExpTorsionAnglePrefs=useExpTorsionAnglePrefs, useBasicKnowledge=useBasicKnowledge, enforceChirality=enforceChirality, numThreads=0)
    mol = Chem.RemoveHs(mol)
    return mol

def get_mol_multipose(mol):
    coords = []
    for n in range(mol.GetNumConformers()):
        coords.append(get_mol_coords(mol, n))
    return MultiPose(coord=torch.stack(coords))


class BigBindStructV2Dataset(Dataset):

    def __init__(self, cfg, split, transform):
        super().__init__(cfg, transform)
        csv = cfg.platform.bigbind_struct_v2_dir + f"/structures_{split}.csv"
        self.structures = pd.read_csv(csv)

        max_residues = self.cfg.data.get("max_rec_residues", None)
        if max_residues is not None:
            self.structures = self.structures.query("num_pocket_residues <= @max_residues").reset_index(drop=True)

        self.dir = cfg.platform.bigbind_struct_v2_dir
        self.split = split
        self.rec_prefix = self.cfg.data.dock_strategy

    @staticmethod
    def get_name():
        return "bigbind_struct_v2"

    def len_impl(self):
        return len(self.structures)

    def get_lig_crystal_file(self, index):
        """ returns the first lig file if use_lig is false, to ensure
        that all ligs are the same """
        if not self.cfg.data.use_lig:
            return self.dir + "/ELNE_HUMAN_30_247_0/3q77_2hy_lig.sdf"
        return self.dir + "/" + self.structures.lig_crystal_file[index]

    def get_lig_uff_file(self, index):
        """ returns the first lig file if use_lig is false, to ensure
        that all ligs are the same """
        if not self.cfg.data.use_lig:
            return self.dir + "/ELNE_HUMAN_30_247_0/3q77_2hy_lig_uff.sdf"
        return self.dir + "/" + self.structures.lig_uff_file[index]

    def get_rec_file(self, index):
        """ same as above """
        if self.cfg.data.use_rec:
            poc_file = self.structures[f"{self.rec_prefix}_rec_pocket_file"][index]
            rec_file = self.structures[f"{self.rec_prefix}_rec_file"][index]
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

        lig_crystal = get_mol_from_file_no_cache(self.get_lig_crystal_file(index))
        lig_crystal = Chem.RemoveHs(lig_crystal)
        lig_crystal_pose = Pose(get_mol_coords(lig_crystal, 0))

        if self.cfg.data.get("use_embed_crystal_pose", False):

            if self.cfg.data.get("direct_align_embed_pose", False):
                lig = get_mol_from_file(self.get_lig_uff_file(index))
                lig = Chem.RemoveHs(lig)
                order = lig.GetSubstructMatch(lig_crystal)
                lig = Chem.RenumberAtoms(lig, list(order))

                lig_torsion_data = TorsionData.from_mol(lig)
                embed_pose = Pose(get_mol_coords(lig, 0))
                lig_embed_crystal_pose = align_poses(embed_pose, lig_crystal_pose, lig_torsion_data)

            else:

                lig = Chem.MolFromSmiles(self.structures.lig_smiles[index])
                order = lig.GetSubstructMatch(lig_crystal)
                lig = Chem.RenumberAtoms(lig, list(order))

                lig = gen_conformers(lig, self.cfg.data.num_poses)
                embed_poses = get_mol_multipose(lig)
                rmsds = get_rmsds([lig], [embed_poses], [lig_crystal_pose], align=True)[0]
                best_conf = lig.GetConformer(rmsds.argmin().item())
                lig_embed_crystal = deepcopy(lig)
                lig_embed_crystal.RemoveAllConformers()
                lig_embed_crystal.AddConformer(best_conf, True)
                Chem.rdMolAlign.AlignMol(lig_embed_crystal, lig_crystal)
                lig_embed_crystal_pose = Pose(get_mol_coords(lig_embed_crystal, 0))
        else:
            lig = get_mol_from_file(self.get_lig_uff_file(index))
            lig = Chem.RemoveHs(lig)
            order = lig.GetSubstructMatch(lig_crystal)
            lig = Chem.RenumberAtoms(lig, list(order))
            lig_embed_crystal_pose = lig_crystal_pose
        

        rec = get_prot_from_file(rec_file)
        poc_id = self.structures.pocket[index]

        x = DFRow(lig=lig, lig_crystal=lig_crystal, rec=rec, pocket_id=poc_id)
        
        y = DFRow(lig_crystal_pose=lig_crystal_pose, lig_embed_crystal_pose=lig_embed_crystal_pose)

        return x, y