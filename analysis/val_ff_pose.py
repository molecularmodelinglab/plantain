from common.cfg_utils import get_config
from common.wandb_utils import get_old_model
from models.model import RandomPoseScore
from models.sym_diffusion import SymDiffusion
from validation.validate import validate
from common.utils import flatten_dict

cfg = get_config("gnina_ff")
cfg.data.pose_sample="best"
# model = RandomPoseScore()
# model.cache_key = "random"
model = get_old_model(cfg, "thicc_4", "best_k")
metrics, plots = validate(cfg, model, "bigbind_gnina_struct", "val", None)
for key, val in flatten_dict(metrics).items():
    print(f"{key}: {val:.3f}")