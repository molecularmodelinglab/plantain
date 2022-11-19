import pandas as pd
import torch
from rdkit import Chem
from common.utils import get_mol_from_file, get_prot_from_file

from datasets.cacheable_dataset import CacheableDataset
from datasets.data_types import InteractionActivityData
from datasets.graphs.interaction_graph import InteractionGraph

class BigBindVinaDataset(CacheableDataset):

    def __init__(self, cfg, split):
        super().__init__(cfg, "bigbind_vina")
        csv = cfg.platform.bigbind_vina_dir + f"/activities_sna_1_{split}.csv"
        self.activities = pd.read_csv(csv)
        self.dir = cfg.platform.bigbind_dir
        self.vina_dir = cfg.platform.bigbind_vina_dir
        self.cfg = cfg

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

    def get_item_pre_cache(self, index):

        lig_file = self.get_lig_file(index)
        rec_file = self.get_rec_file(index)

        lig = get_mol_from_file(lig_file)
        lig = Chem.RemoveHs(lig)
        rec = get_prot_from_file(rec_file)

        is_active = torch.tensor(self.activities.active[index], dtype=bool)

        inter_graph = InteractionGraph(self.cfg, lig, rec)
        ret = InteractionActivityData(inter_graph, is_active)

        return ret

    def get_variance(self):
        return {}

    def get_type_data(self):
        return InteractionActivityData.get_type_data(self.cfg)