import importlib
import numpy as np
import torch
import random
from datasets.base_datasets import Dataset
from datasets.combo_dataloader import ComboDataloader
from terrace import DataLoader

# import all the files in the directory so we can create
# a name to dataset mapping
import os
for module in os.listdir(os.path.dirname(__file__)):
    if module == '__init__.py' or module[-3:] != '.py':
        continue
    importlib.import_module('datasets.'+module[:-3])

def flatten_subclasses(cls):
    ret = [ cls ]
    for subclass in cls.__subclasses__():
        ret += flatten_subclasses(subclass)
    return ret

name_to_dataset = {}
for class_ in flatten_subclasses(Dataset):
    try:
        name_to_dataset[class_.get_name()] = class_
    except NotImplementedError:
        pass

SEED = 49
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def make_dataset(cfg, name, split, transform):
    ret = name_to_dataset[name](cfg, split, transform)

    # In order for pickle to work on the results from this dataset,
    # it must know the types of the data (which are created dynamically).
    # Thus, we must ensure that the return types are created in the main
    # thread before the dataset is used in any other thread.
    # This is why we create the first item in the dataset and do
    # nothing with it

    ret[0]

    return ret


def make_dataloader(cfg, name, split, transform, force_no_shuffle=False):
    dataset = make_dataset(cfg, name, split, transform)
    n_workers = cfg.platform.num_workers
    if force_no_shuffle:
        shuffle = False
    else:
        shuffle = (split == "train")
    return DataLoader(dataset,
                      batch_size=cfg.batch_size,
                      num_workers=n_workers,
                      pin_memory=True,
                      shuffle=shuffle,
                      worker_init_fn=seed_worker)

def make_train_dataloader(cfg, transform, force_no_shuffle=False):
    if isinstance(cfg.train_dataset, str):
        return make_dataloader(cfg, cfg.train_dataset, "train", transform, force_no_shuffle)
    else:
        assert "combo" in cfg.train_dataset
        loaders = []
        for name in cfg.train_dataset.combo:
            loaders.append(make_dataloader(cfg, name, "train", transform, force_no_shuffle=force_no_shuffle))
        return ComboDataloader(loaders)

