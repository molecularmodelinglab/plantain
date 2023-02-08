from glob import glob
import pickle
import torch
import pandas as pd
from common.cfg_utils import get_config
from common.utils import flatten_dict

def main(cfg):
    folders = {
        "cLogP x Diam": "wandb:hpte2jh3:v8",
        "cLogP": "wandb:hpte2jh3:v10",
        "Predicting own loss": "wandb:05nsdfh5:v3",
        "Predicting GNINA score": "wandb:nvnc5i6c:v3"
    }
    
    rows = []
    for name, folder in folders.items():
        for type_, pre in (("(Reject)", "rf_"), ("(Regular)", "")):
            full_name = name + " " + type_
            full_folder = "outputs/results/" + pre + folder
            for target in glob(full_folder + "/lit_pcba*"):
                with open(target + "/metrics.pkl", "rb") as f:
                    metrics = flatten_dict(pickle.load(f))
                cur_row = { "model": full_name, "target": target.split("lit_pcba_")[-1]}
                for key, val in metrics.items():
                    if key.startswith("select"): continue
                    if isinstance(val, torch.Tensor):
                        val = val.cpu().item()
                    cur_row[key] = val
                rows.append(cur_row)
    df = pd.DataFrame(rows)
    for name in df.model.unique():
        cur_df = df.query("model == @name").reset_index(drop=True)
        med_ef1 = cur_df.enrichment_ef1.median()
        mean_ef1 = cur_df.enrichment_ef1.mean()
        print(f"{name} LIT-PCBA metrics:")
        print(f"  Median EF1%: {med_ef1:.2f}")
        print(f"  Mean EF1%: {mean_ef1:.2f}")

        

if __name__ == "__main__":
    cfg = get_config("attention_gnn")
    main(cfg)