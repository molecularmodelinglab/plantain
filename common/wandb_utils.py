import wandb
import os

from common.cfg_utils import get_run_config
from training.trainer import Trainer

_api = wandb.Api()

def get_weight_artifact(run, tag):
    return _api.artifact(f"{run.project}/model-{run.id}:{tag}", type='model')

def get_old_model(cfg, run_name, tag="latest"):
    """ Get the wandb run with name run_name, and return a model
    loaded from the saved checkpoint with the correct tag"""
    runs = _api.runs(path=cfg.project, filters={"display_name": run_name})
    assert len(runs) == 1
    run = runs[0]
    cfg = get_run_config(run, cfg)
    artifact = get_weight_artifact(run, tag)
    artifact_dir = f"artifacts/model-{run.id}:{artifact.version}"
    if not os.path.exists(artifact_dir):
        assert os.path.normpath(artifact_dir) == os.path.normpath(artifact.download())
    checkpoint_file = artifact_dir + "/model.ckpt"
    trainer = Trainer.from_checkpoint(cfg, checkpoint_file)
    return trainer.model