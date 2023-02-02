import pickle
from common.cfg_utils import get_config
from common.wandb_utils import get_old_model
from validation.validate import save_validation
from datasets.lit_pcba import LitPcbaDataset

def main(cfg):
    model_names = [ "bce_mse" ]
    for name in model_names:
        model = get_old_model(cfg, name, "latest")
        dataset = "lit_pcba"
        for target in LitPcbaDataset.get_all_targets(cfg): 
            print(f"Validating {name} on {dataset}_{target}")
            save_validation(cfg, model, dataset, target, 10)

if __name__ == "__main__":
    cfg = get_config("attention_gnn")
    main(cfg)
