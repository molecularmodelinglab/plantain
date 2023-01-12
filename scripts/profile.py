from datetime import datetime
import sys
import cProfile
import traceback
from models.make_model import make_model
from common.cfg_utils import get_config
from training.trainer import Trainer
from validation.validate import validate

def profile(task):

    stamp = datetime.timestamp(datetime.now())
    out_fname = f"outputs/{task}-{stamp}.prof"

    if task == "validate":
        cfg = get_config("attention_gnn")
        model = make_model(cfg)
        fn = lambda: validate(cfg, model, cfg.val_datasets[0], "val", 200)
    elif task == "train":
        cfg = get_config("attention_gnn")
        cfg.profile_max_batches = 250
        trainer = Trainer(cfg)
        fn = lambda: trainer.fit(None, [])
    else:
        raise ValueError(f"Invalid profile task {task}")

    pr = cProfile.Profile()
    pr.enable()
    try:
        fn()
    except:
        traceback.print_exc()
    pr.disable()
    print(f"Saving profile to {out_fname}")
    pr.dump_stats(out_fname)

if __name__ == "__main__":
    profile(sys.argv[1])