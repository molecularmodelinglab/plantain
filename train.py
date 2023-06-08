#!/usr/bin/env python

import sys
import warnings

# needed because dgllife can't find rdkit otherwise...
from rdkit import Chem

import resource

from training.trainer import Trainer
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (32768, rlimit[1]))

import os
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from common.cfg_utils import get_config, get_run_config

def train(cfg):

    callbacks = []
    if cfg.project is not None:
        project = cfg.project

        logger = WandbLogger(project=project, name=cfg.name, log_model="all")
        logger.log_hyperparams(cfg)
        if "SLURM_JOB_ID" in os.environ:
            logger.log_hyperparams({ "slurm_job_id": os.environ["SLURM_JOB_ID"] })

        checkpoint_callback = ModelCheckpoint(monitor=cfg.monitor_metric.name, mode=cfg.monitor_metric.mode, save_last=True, every_n_epochs=1)
        callbacks.append(checkpoint_callback)
    else:
        logger = None
        
    trainer = Trainer(cfg)
    trainer.fit(logger, callbacks)


if __name__ == "__main__":
    if len(sys.argv) > 1 and '=' not in sys.argv[1]:
        cfg_name = sys.argv[1]
    else:
        raise AssertionError("First argument must be name of config profile!")
    cfg = get_config(cfg_name=cfg_name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train(cfg)
