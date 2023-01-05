import pickle
import os
import uuid
import torch
from traceback import print_exc
from torch.utils import data
from omegaconf import OmegaConf
from random import random

from datasets.utils import dict_to_id_str

CACHE_VERSION = 0.6

def get_dataset_hash(cfg):
    cfg_dict = OmegaConf.to_container(cfg.data)
    dict_str = dict_to_id_str(cfg_dict)
    dict_id = uuid.uuid3(uuid.NAMESPACE_DNS, dict_str)
    return dict_id.hex

class CacheableDataset(data.Dataset):

    def __init__(self, cfg, dataset_name):
        super().__init__()
        self.cache_postfix = get_dataset_hash(cfg)
        self.cache = cfg.data.cache
        self.cache_dir = cfg.platform.cache_dir + "/" + dataset_name
        os.makedirs(self.cache_dir, exist_ok=True)

    def __getitem__(self, index):

        if index >= len(self):
            raise StopIteration

        random_works = True
        r = random()
        try:
            key = self.get_randomized_cache_key(index, r)
        except NotImplementedError:
            random_works = False
        if not random_works:
            key = self.get_cache_key(index)
        
        if self.cache:
            cache_file = f"{self.cache_dir}/{CACHE_VERSION}/{self.cache_postfix}/{key}_cache.pkl"
            cache_folder = "/".join(cache_file.split("/")[:-1])
            os.makedirs(cache_folder, exist_ok=True)

            try:
                with open(cache_file, "rb") as f:
                    batch = pickle.load(f)
                    return batch
            except KeyboardInterrupt:
                raise
            except:
                # raise
                pass

        random_works = True
        try:
            ret = self.get_randomized_item_pre_cache(index, r)
        except NotImplementedError:
            random_works = False
        if not random_works:
            ret = self.get_item_pre_cache(index)
    
        if self.cache:
            try:
                with open(cache_file, "wb") as f:
                    pickle.dump(ret, f)
            except KeyboardInterrupt:
                raise
            except:
                print(f"Failed to cache item at index {index} into {cache_file}...")
                print_exc()

        return ret

    def get_randomized_cache_key(self, index, r):
        raise NotImplementedError
                   
    def get_cache_key(self, index):
        raise NotImplementedError

    def get_item_pre_cache(self, index):
        raise NotImplementedError

    def get_randomized_item_pre_cache(self, index, r):
        raise NotImplementedError
