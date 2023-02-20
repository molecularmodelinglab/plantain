from common.cfg_utils import get_config
from common.wandb_utils import get_old_model
from validation.validate import validate

cfg = get_config("diffusion")
cfg.batch_size=1
model = get_old_model(cfg, "intra_lig_energy", "best_k")
metrics, plots = validate(cfg, model, cfg.val_datasets[0], "val", 50)
for key, val in metrics.items():
    print(f"{key}: {val:.3f}")