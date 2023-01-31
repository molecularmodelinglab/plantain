import pickle
from common.cfg_utils import get_config
from common.wandb_utils import get_old_model
from validation.validate import save_validation

def main(cfg):
    model_names = [ "pos_softmax_0.5", "softmax_0.001", "select_correct", "bce_mse",]# "combo_1.5_0.5" ]
    for name in model_names:
        print(f"Validating {name}")
        model = get_old_model(cfg, name, "latest")
        save_validation(cfg, model, "bigbind_act", "val", 250)

if __name__ == "__main__":
    cfg = get_config("attention_gnn")
    main(cfg)
