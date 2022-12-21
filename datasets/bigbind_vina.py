import os
import pickle
import multiprocessing
import tarfile
import warnings
import pandas as pd
import torch
import tempfile
from tqdm import tqdm
from rdkit import Chem
from common.utils import get_mol_from_file, get_prot_from_file

from Bio.PDB.mmtf import MMTFParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning

from datasets.cacheable_dataset import CacheableDataset
from datasets.data_types import InteractionActivityData
from datasets.graphs.interaction_graph import InteractionGraph
from datasets.graphs.mol_graph import MolGraph
from datasets.graphs.prot_graph import ProtGraph

class BigBindVinaDataset(CacheableDataset):

    def __init__(self, cfg, split):
        super().__init__(cfg, "bigbind_vina")
        csv = cfg.platform.bigbind_vina_dir + f"/activities_sna_1_{split}.csv"
        self.activities = pd.read_csv(csv)
        self.dir = cfg.platform.bigbind_dir
        self.vina_dir = cfg.platform.bigbind_vina_dir
        self.cfg = cfg
        self.split = split
        self.tars = {}

    def __len__(self):
        return len(self.activities)

    # smh a lot of this is brazenly copy-and-pasted from bigbind_act
    def get_lig_file(self, index):
        """ returns the first lig file if use_lig is false, to ensure
        that all ligs are the same """
        if not self.cfg.data.use_lig:
            return self.vina_dir + "/train/0.pdbqt"
        return self.vina_dir + "/" + self.activities.docked_lig_file[index]

    def get_rec_file(self, index):
        """ same as above """
        if self.cfg.data.use_rec:
            poc_file = self.activities.ex_rec_pocket_file[index]
            rec_file = self.activities.ex_rec_file[index]
        else:
            poc_file = "ELNE_HUMAN_30_247_0/3q77_A_rec_pocket.pdb"
            rec_file = "ELNE_HUMAN_30_247_0/3q77_A_rec.pdb"
        if self.cfg.data.rec_graph.only_pocket:
            return self.dir + "/" + poc_file
        else:
            return self.dir + "/" + rec_file

    def get_cache_key(self, index):
        
        lig_file = self.get_lig_file(index).split("/")[-1]
        rec_file = self.get_rec_file(index).split("/")[-1]

        return lig_file + "_" + rec_file

    def get_lig(self, lig_file, tar):
        lig_file = "/".join(lig_file.split("/")[-2:])
        f = tar.extractfile(lig_file)
        # f = self.tar_files[lig_file]
        return pickle.load(f)

    def get_rec(self, rec_file, tar):
        rec_file = "/".join(rec_file.split("/")[-2:])
        f = tar.extractfile(rec_file)
        # f = self.tar_files[rec_file]
        with tempfile.NamedTemporaryFile() as tmp:
            tmp.write(f.read())
            tmp.seek(0, os.SEEK_SET)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=PDBConstructionWarning)
                structure = MMTFParser().get_structure(tmp.name)
        return structure[0]

    def get_item_pre_cache(self, index):

        if self.cfg.data.use_tar:
            proc = multiprocessing.current_process()
            if proc not in self.tars:
                self.tars[proc] = tarfile.open(self.cfg.platform.bigbind_vina_dir + f"/{self.split}_files.tar", "r:")
            tar = self.tars[proc]

        lig_file = self.get_lig_file(index)
        rec_file = self.get_rec_file(index)

        try:

            is_active = torch.tensor(self.activities.active[index], dtype=bool)

            if self.cfg.data.use_tar:
                lig = self.get_lig(lig_file, tar)
                rec = self.get_rec(rec_file, tar)
            else:
                lig = get_mol_from_file(lig_file)
                rec = get_prot_from_file(rec_file)

            lig = Chem.RemoveHs(lig)

            rec_graph = ProtGraph(self.cfg, rec)

            inter_graphs = []
            for conformer in range(self.cfg.data.num_conformers):
                if conformer >= lig.GetNumConformers():
                    conformer = 0

                lig_graph = MolGraph(self.cfg, lig, conformer)
                inter_graph = InteractionGraph(self.cfg, lig_graph, rec_graph)
                inter_graphs.append(inter_graph)


            ret = InteractionActivityData(tuple(inter_graphs), is_active)
        except KeyboardInterrupt:
            raise
        except:
            print(f"Error proccessing item at {index=}")
            print(f"{lig_file=}")
            print(f"{rec_file=}")
            raise

        return ret

    def get_variance(self):
        return {}

    def get_type_data(self):
        return InteractionActivityData.get_type_data(self.cfg)
