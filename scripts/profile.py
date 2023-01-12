from datetime import datetime
import sys
import cProfile
from models.make_model import make_model
from common.cfg_utils import get_config
from validation.validate import validate

def profile(task):

    stamp = datetime.timestamp(datetime.now())
    out_fname = f"outputs/{task}-{stamp}.prof"
    print(f"Saving profile to {out_fname}")

    if task == "validate":
        cfg = get_config("attention_gnn")
        model = make_model(cfg)
        cProfile.runctx('validate(cfg, model, cfg.val_datasets[0], "val", 100)', globals(), locals(), out_fname)
    else:
        raise ValueError(f"Invalid profile task {task}")

if __name__ == "__main__":
    profile(sys.argv[1])