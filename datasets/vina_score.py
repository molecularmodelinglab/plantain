import sys
sys.path.insert(0, './terrace')

import torch
import numpy as np
from torch.utils import data
import random
import pandas as pd
from glob import glob
from vina import Vina
from traceback import print_exc
from meeko import MoleculePreparation, PDBQTMolecule
import rdkit
from rdkit import Chem
from rdkit.Chem.rdShapeHelpers import ComputeConfBox, ComputeUnionBox


from datasets.graphs.mol_graph import MolGraph, mol_graph_from_sdf
from datasets.graphs.prot_graph import ProtGraph, prot_graph_from_pdb
from datasets.data_types import EnergyData
from common.utils import rand_rotation_matrix


split_fracs = {
    "train": (0.0, 0.7),
    "val": (0.7, 0.85),
    "test": (0.85, 1.0)
}

def move_lig_to_center(cfg, lig, center):
    center_pt = rdkit.Geometry.rdGeometry.Point3D(*center)
    conf = lig.GetConformer()
    # mat = rand_rotation_matrix()
    # print(Chem.MolToSmiles(lig))
    # print(conf.GetNumAtoms())
    # Chem.rdMolTransforms.TransformConformer(conf, mat)
    lig_center = Chem.rdMolTransforms.ComputeCentroid(conf)
    rand_pt = rdkit.Geometry.rdGeometry.Point3D(*(torch.randn((3,))*cfg.data.lig_trans_dist).tolist())
    for i in range(lig.GetNumAtoms()):
        og_pos = conf.GetAtomPosition(i)
        new_pos = og_pos + center_pt - lig_center + rand_pt
        conf.SetAtomPosition(i, new_pos)

def get_bounds(ligs, padding=6):
    bounds = None
    for lig in ligs:
        box = ComputeConfBox(lig.GetConformer(0))
        if bounds is None:
            bounds = box
        else:
            bounds = ComputeUnionBox(box, bounds)

    bounds_min = np.array([bounds[0].x, bounds[0].y, bounds[0].z])
    bounds_max = np.array([bounds[1].x, bounds[1].y, bounds[1].z])
    center = 0.5*(bounds_min + bounds_max)
    size = (bounds_max - center + padding)*2
    return tuple(center), tuple(size)

class VinaScoreDataset(data.Dataset):
    
    def __init__(self, cfg, split):
        super().__init__()
        rec_files = glob(cfg.platform.bigbind_docked_dir + "/*/*_rec_pocket.pdbqt")
        lig_files = glob(cfg.platform.bigbind_dir + "/chembl_structures/*.sdf")
        frac_start, frac_end = split_fracs[split]
        self.cfg = cfg
        self.rec_files = rec_files[int(frac_start*len(rec_files)):int(frac_end*len(rec_files))]
        self.lig_files = lig_files[int(frac_start*len(lig_files)):int(frac_end*len(lig_files))]
        self.struct_df = pd.read_csv(cfg.platform.bigbind_dir + "/structures_all.csv")

    def __len__(self):
        return len(self.lig_files)

    def __getitem__(self, index):
        try:
            if index >= len(self):
                raise StopIteration
            lig_file = self.lig_files[index]
            rec_file = random.choice(self.rec_files)

            rec_pdb = rec_file.split("/")[-1].split("_")[0]
            rec_data = self.struct_df.query("ex_rec_pdb == @rec_pdb").iloc[0]
            center = (rec_data.pocket_center_x, rec_data.pocket_center_y, rec_data.pocket_center_z)
            size = (rec_data.pocket_size_x, rec_data.pocket_size_y, rec_data.pocket_size_z)
            rec_pdb_file = self.cfg.platform.bigbind_dir + "/" + rec_data.ex_rec_pocket_file

            lig = next(Chem.SDMolSupplier(lig_file, sanitize=True))
            move_lig_to_center(self.cfg, lig, center)
            _, size = get_bounds([lig])

            preparator = MoleculePreparation(hydrate=False) # macrocycles flexible by default since v0.3.0
            preparator.prepare(lig)
            lig_str = preparator.write_pdbqt_string()

            v = Vina(sf_name='vina', verbosity=0, cpu=1, no_refine=True)

            v.set_receptor(rec_file)
            v.set_ligand_from_string(lig_str)
            v.compute_vina_maps(center=center, box_size=size)
            # v.randomize()

            energy = v.score()[0]
            lig_graph = mol_graph_from_sdf(self.cfg, lig_file)
            rec_graph = prot_graph_from_pdb(self.cfg, rec_file)

            return EnergyData(lig_graph, rec_graph, energy)

        except KeyboardInterrupt:
            raise
        except:
            # raise
            print(f"Error proccessing item at {index=}")
            print(f"{lig_file=}")
            print(f"{rec_file=}")
            print_exc()
            new_idx = int(random.random())*len(self)
            return self[new_idx]

    def get_variance(self):
        # computed from 800 datapoints (below)
        return {
            "energy": 4651.33251953125,
        }

    def get_type_data(self):
        return EnergyData.get_type_data(self.cfg)

if __name__ == "__main__":
    from common.cfg_utils import get_config
    from datasets.make_dataset import make_dataloader
    from tqdm import tqdm
    cfg = get_config("./configs", "vina_score")
    cfg.batch_size = 8
    energies = []
    # dataset = VinaScoreDataset(cfg, "val")
    # print(len(dataset))
    # for d in dataset:
    #     print(d.energy)
    loader = make_dataloader(cfg, "val")
    energies = []
    tot = 0
    amount_to_test = 100
    for batch in tqdm(loader,total=amount_to_test):
        energies += batch.energy.tolist()
        tot += 1
        if tot > amount_to_test:
            break
    print(f"Var[U] = {torch.tensor(energies).var()}")