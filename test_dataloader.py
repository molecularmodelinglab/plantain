import sys
# sys.path = ['', '/opt/conda/lib/python38.zip', '/opt/conda/lib/python3.8', '/opt/conda/lib/python3.8/lib-dynload', '/opt/conda/lib/python3.8/site-packages', '/opt/conda/lib/python3.8/site-packages/torchtext-0.11.0a0-py3.8-linux-x86_64.egg']
sys.path.insert(0, './terrace')

# needed because dgllife is stupid and can't find rdkit otherwise...
from rdkit import Chem
from tqdm import tqdm

# this is needed because otherwise pytorch dataloaders will just fail
# https://github.com/pytorch/pytorch/issues/973
# import torch
# torch.multiprocessing.set_sharing_strategy('file_system')

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

import os
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from routines.ai_routine import AIRoutine
from common.cfg_utils import get_config, get_run_config

def test_dataloader(cfg):
    routine = AIRoutine(cfg)
    print(cfg.data.sna_frac)
    for batch in tqdm(routine.train_dataloader):
        pass

if __name__ == "__main__":
    ffg_name = "default"
    cfg = get_config(cfg_name="default")
    test_dataloader(cfg)
