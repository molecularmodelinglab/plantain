import sys
# sys.path = ['', '/opt/conda/lib/python38.zip', '/opt/conda/lib/python3.8', '/opt/conda/lib/python3.8/lib-dynload', '/opt/conda/lib/python3.8/site-packages', '/opt/conda/lib/python3.8/site-packages/torchtext-0.11.0a0-py3.8-linux-x86_64.egg']
sys.path.insert(0, './terrace')

# needed because dgllife is stupid and can't find rdkit otherwise...
from rdkit import Chem


# this is needed because otherwise pytorch dataloaders will just fail
# https://github.com/pytorch/pytorch/issues/973
# import torch
# torch.multiprocessing.set_sharing_strategy('file_system')

# import resource
# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

from routines.ai_routine import AIRoutine
from common.cfg_utils import get_config, get_run_config

from tqdm import tqdm
import pytorch_lightning as pl

if __name__ == "__main__":


    if len(sys.argv) > 1 and '=' not in sys.argv[1]:
        cfg_name = sys.argv[1]
    else:
        cfg_name = "default"
    cfg = get_config(cfg_name=cfg_name)
    cfg.project = None
    cfg.platform.num_workers=12
    cfg.val_check_interval = 1

    routine = AIRoutine(cfg)
    for batch in tqdm(routine.train_dataloader):
        pass


    # routine.fit(None, [])
    trainer = pl.Trainer(gpus=1, val_check_interval=cfg.val_check_interval)
    trainer.fit(routine,
                routine.train_dataloader,
                routine.val_dataloader)
    trainer.validate(routine, routine.val_dataloader)