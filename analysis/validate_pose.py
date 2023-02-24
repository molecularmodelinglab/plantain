from common.cfg_utils import get_config
from common.wandb_utils import get_old_model
from models.sym_diffusion import SymDiffusion
from validation.validate import validate

cfg = get_config("diff_combo")
# cfg.batch_size=1
# model = SymDiffusion(cfg)
model = get_old_model(cfg, "combo", "latest")
metrics, plots = validate(cfg, model, cfg.val_datasets[0], "val", None)
for key, val in metrics.items():
    print(f"{key}: {val:.3f}")