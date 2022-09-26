import pandas as pd

from datasets.cacheable_dataset import CacheableDataset

class BigBindDataset(CacheableDataset):

    def __init__(self, cfg, name, split):
        super(BigBindDataset, self).__init__(cfg, name)
        self.cfg = cfg
        self.dir = cfg.platform.bigbind_dir
        all_structures = pd.read_csv(self.dir + f"/structures_{split}.csv")
        all_activities = pd.read_csv(self.dir + f"/activities_{split}.csv")
        # can delete this once I remove the single atom mols in bigbind
        self.structures = all_structures
        self.activities = all_activities[all_activities.lig_smiles.str.len() > 5].reset_index(drop=True)
        max_pocket_size=42
        # todo: we need to filter like this beforehand
        self.activities = self.activities.query("num_pocket_residues >= 5 and pocket_size_x < @max_pocket_size and pocket_size_y < @max_pocket_size and pocket_size_z < @max_pocket_size").reset_index(drop=True)
        if cfg.data.only_ki_kd:
            self.activities = self.activities.query("standard_type == 'Ki' or standard_type == 'Kd' ").reset_index(drop=True)