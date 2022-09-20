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