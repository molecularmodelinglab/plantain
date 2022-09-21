import os
import wandb
# needed because dgllife is stupid and can't find rdkit otherwise...
from rdkit import Chem
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from routines.routine import Routine
from common.cfg_utils import get_config

def train(cfg):

    callbacks = []
    if cfg.project is not None:
        if cfg.resume_id:
            run = wandb.init(project=cfg.project, id=cfg.resume_id, resume=True)
            cfg = get_config(run, cfg)
            artifact = run.use_artifact(f"{cfg.project}/model-{run.id}:latest", type='model')
            artifact_dir = artifact.download()
            checkpoint_file = artifact_dir + "/model.ckpt"
            routine = Routine.from_checkpoint(cfg, checkpoint_file)
        
        logger = WandbLogger(project=cfg.project, name=cfg.name, log_model="all")
        logger.log_hyperparams(cfg)
        if "SLURM_JOB_ID" in os.environ:
            logger.log_hyperparams({ "slurm_job_id": os.environ["SLURM_JOB_ID"] })
        checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")
        callbacks.append(checkpoint_callback)
        routine = Routine(cfg)
    else:
        routine = Routine(cfg)
        logger = None

    routine.fit(logger, callbacks)

if __name__ == "__main__":
    cfg = get_config()
    train(cfg)
