from git_timewarp import GitTimeWarp
import wandb
import os

from routines.ai_routine import AIRoutine
from common.cfg_utils import get_config, get_run_config
from common.cache import cache

def get_weight_artifact(run, tag="latest"):
    api = wandb.Api()
    return api.artifact(f"{run.project}/model-{run.id}:latest", type='model')

def get_old_routine(cfg, run, tag="latest"):
    cfg = get_run_config(run, cfg)
    artifact = get_weight_artifact(run, tag)
    artifact_dir = f"artifacts/model-{run.id}:{artifact.version}"
    if not os.path.exists(artifact_dir):
        assert os.path.normpath(artifact_dir) == os.path.normpath(artifact.download())
    checkpoint_file = artifact_dir + "/model.ckpt"
    routine = AIRoutine.from_checkpoint(cfg, checkpoint_file)
    return routine

def old_model_key(cfg, run, tag):
    artifact = get_weight_artifact(run, tag)
    return (run.id, artifact.version)

@cache(old_model_key)
def get_old_model(cfg, run, tag="latest"):
    routine = get_old_routine(cfg, run, tag)
    return routine.model
