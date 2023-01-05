import importlib
import numpy as np
import torch
import random
from datasets.base_datasets import Dataset
from terrace import DataLoader

# import all the files in the directory so we can create
# a name to dataset mapping
import os
for module in os.listdir(os.path.dirname(__file__)):
    if module == '__init__.py' or module[-3:] != '.py':
        continue
    importlib.import_module('datasets.'+module[:-3])

name_to_dataset = {}
for class_ in Dataset.__subclasses__():
    name_to_dataset[class_.get_name()] = class_

SEED = 49
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def make_dataset(cfg, split, transform):
    return name_to_dataset[cfg.dataset](cfg, split, transform)

def make_dataloader(cfg, split, transform, force_no_shuffle=False):
    dataset = make_dataset(cfg, split, transform)
    # todo: change back once we've solved the infamous dgl bug
    n_workers = cfg.platform.num_workers#  if split == "train" else 0
    if force_no_shuffle:
        shuffle = False
    else:
        shuffle = (split == "train")
    return DataLoader(dataset,
                      batch_size=cfg.batch_size,
                      num_workers=n_workers,
                      pin_memory=True,
                      # sampler=sampler,
                      shuffle=shuffle,
                      worker_init_fn=seed_worker)