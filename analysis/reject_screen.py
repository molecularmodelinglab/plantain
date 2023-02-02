import os
import pickle
from common.cfg_utils import get_config
from common.wandb_utils import get_old_model
from models.gp_uncertainty import GPUncertainty
from validation.validate import save_validation
from datasets.lit_pcba import LitPcbaDataset

def main(cfg):
    model_names = [ "bce_mse" ]
    for name in model_names:
        model = get_old_model(cfg, name, "latest")
        u_model = GPUncertainty(cfg, model)
        u_model.fit("bigbind_act", "val", 500)

        out_folder =f"outputs/results/{u_model.cache_key}"
        print(f"Saving gp plot to {out_folder}")
        os.makedirs(out_folder, exist_ok=True)
        fig = u_model.plot()
        fig.savefig(f"{out_folder}/gp_plot.pdf")

        dataset = "lit_pcba"
        for target in LitPcbaDataset.get_all_targets(cfg): 
            print(f"Validating {name} on {dataset}_{target}")
            save_validation(cfg, u_model, dataset, target, None)
            save_validation(cfg, model, dataset, target, None)

if __name__ == "__main__":
    cfg = get_config("attention_gnn")
    main(cfg)
