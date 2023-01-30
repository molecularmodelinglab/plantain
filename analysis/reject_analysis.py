import pickle
from common.cfg_utils import get_config
from common.wandb_utils import get_old_model
from validation.validate import validate

def main(cfg):
    model = get_old_model(cfg, "softmax_0.001", "latest")
    ret = validate(cfg, model, "bigbind_act", "val", None)
    fname = "outputs/reject_select_softmax_metrics.pkl"
    print(f"Saving to {fname}")
    with open(fname, 'wb') as f:
        pickle.dump(ret, f)
    with open(fname, "rb") as f:
        print(pickle.load(f))

if __name__ == "__main__":
    cfg = get_config("attention_gnn")
    main(cfg)
