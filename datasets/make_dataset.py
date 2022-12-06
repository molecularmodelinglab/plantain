import numpy as np
import torch
from torch.utils.data import DistributedSampler
import random

from datasets.bigbind_act import BigBindActDataset
from datasets.bigbind_struct import BigBindStructDataset
from datasets.bigbind_fp import BigBindFpDataset
from datasets.bigbind_test import BigBindTestDataset
# from datasets.vina_score import VinaScoreDataset
from terrace.batch import DataLoader

from datasets.bigbind_vina import BigBindVinaDataset

SEED = 49
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def make_dataset(cfg, split):
    return {
        "bigbind_act": BigBindActDataset,
        "bigbind_struct": BigBindStructDataset,
        "bigbind_fp": BigBindFpDataset,
        "bigbind_test": BigBindTestDataset,
        "bigbind_vina": BigBindVinaDataset,
        # "vina_score": VinaScoreDataset,
    }[cfg.dataset](cfg, split)

def make_dataloader(cfg, split, force_no_shuffle=False):
    dataset = make_dataset(cfg, split)
    n_workers = cfg.platform.num_workers
    if force_no_shuffle:
        shuffle = False
    else:
        shuffle = (split == "train")
    # if split == "train":
    #     sampler = DistributedSampler(dataset, shuffle=shuffle)
    # else:
    #     sampler = None
    return DataLoader(dataset,
                      batch_size=cfg.batch_size,
                      num_workers=n_workers,
                      pin_memory=True,
                      # sampler=sampler,
                      shuffle=shuffle,
                      worker_init_fn=seed_worker)

if __name__ == "__main__":
    from common.cfg_utils import get_config
    cfg = get_config()
    cfg.data.cache = False
    loader = make_dataloader(cfg, "val")
    print(next(iter(loader)).lig)
