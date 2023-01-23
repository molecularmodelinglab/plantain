from common.cfg_utils import get_config
from common.wandb_utils import get_old_model
from models.attention_gnn_pose import AttentionGNNPose
from models.gnina import Gnina
from models.model import RandomPoseScore
from validation.validate import validate

cfg = get_config("attention_gnn")
cfg.data.sna_frac = None
# gnina = Gnina(0)
gnina = AttentionGNNPose(cfg, get_old_model(cfg, "residue_classify", "latest"))
metrics = validate(cfg, gnina, "bigbind_gnina_struct", "val", None)
for key, val in metrics.items():
    print(f"{key}: {val:.3f}")