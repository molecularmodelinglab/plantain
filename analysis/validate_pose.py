from common.cfg_utils import get_config
from common.wandb_utils import get_old_model
from models.sym_diffusion import SymDiffusion
from validation.validate import validate
from common.utils import flatten_dict

cfg = get_config("diffusion")
# cfg.batch_size=1
# model = SymDiffusion(cfg)
model = get_old_model(cfg, "more_data_10", "best_k")
# metrics, plots = validate(cfg, model, cfg.val_datasets[0], "val", 5)
metrics, plots = validate(cfg, model, "bigbind_struct_v2", "val", None)
for key, val in flatten_dict(metrics).items():
    print(f"{key}: {val:.3f}")