
import pandas as pd
import torch
from common.cfg_utils import get_config
from common.utils import flatten_dict
from datasets.make_dataset import make_dataset
from models.diffdock import DiffDock, get_diffdock_indexes
from models.gnina import GninaPose
from models.pretrained_plantain import get_pretrained_plantain
from models.vina import VinaPose
from validation.validate import validate

def get_model(name, split):
    cfg = get_config("icml")
    if name == "plantain":
        model = get_pretrained_plantain()
    if name == "gnina":
        cfg.data.num_poses = 9
        model = GninaPose(cfg)
    elif name == "vina":
        cfg.data.num_poses = 9
        model = VinaPose(cfg)
    elif name == "diffdock":
        cfg.data.num_poses = 40
        model = DiffDock(cfg, split)
    elif name == "combo":
        raise NotImplementedError
    
    return cfg, model

def eval_model_on_crossdocked(name, split, subset):
    print(f"Evaluating {name} on CrossDocked {split} set (subset={subset})")

    num_preds = None
    shuffle_val = False
    timing = True
    dataset_name = "crossdocked"

    cfg, model = get_model(name, split)

    if subset is None:
        subset_indexes = None
    elif subset == "diffdock":
        dataset = make_dataset(cfg, dataset_name, split, [])
        subset_indexes = get_diffdock_indexes(cfg, dataset)

    # prefix = "" if subset is None else "_" + subset
    metrics, (x, y, p, runtimes) = validate(cfg, model, dataset_name, split, num_preds, shuffle_val, subset_indexes, timing)

    if name in ("vina", "gnina"):
        runtimes = []
        runtime_file = f"outputs/{name}_{dataset_name}_structures_{split}_runtimes.txt"
        with open(runtime_file, "r") as f:
            for line in f.readlines():
                runtimes.append(float(line.split(",")[-1]))

        if subset_indexes is not None:
            runtimes = torch.asarray(runtimes)[torch.asarray(list(subset_indexes))]

    out_metrics = { "name": name, "subset": subset }
    out_metrics["subset"] = "none" if subset is None else subset
    for key, val in flatten_dict(metrics).items():
        out_metrics[key] = float(val)

    if name == "diffdock":
        time_file = cfg.platform.diffdock_dir + f"/diffdock_timer_{dataset_name}_{split}.txt"
        with open(time_file, "r") as f:
            content = f.read()

        hhmmss = content.split("elapsed")[0].split(" ")[-1]
        h, m, s = hhmmss.split(":")
        total_runtime = int(h)*60*60 + int(m)*60 + int(s)
        out_metrics["mean_runtime"] = total_runtime/len(subset_indexes)

    else:
        out_metrics["mean_runtime"] = float(sum(runtimes)/len(runtimes))

    return out_metrics


def main():
    split = "test"
    subset_and_models = {
        None: [
            "vina",
            "gnina",
            # "plantain",
        ],
        "diffdock": [
            "vina",
            "gnina",
            "diffdock",
            # "plantain",
        ]
    }

    data = []

    for subset, model_names in subset_and_models.items():
        for name in model_names:
            data.append(eval_model_on_crossdocked(name, split, subset))

    df = pd.DataFrame(data)

    for row in df.itertuples():
        print(f"{row.name} (subset={row.subset})")
        print(f"  <2 Å acc:")
        print(f"    mean: {row.acc_2_mean_1*100:.1f}%, unnorm: {row.acc_2_all_1*100:.1f}%")
        print(f"  <5 Å acc:")
        print(f"    mean: {row.acc_5_mean_1*100:.1f}%, unnorm: {row.acc_5_all_1*100:.1f}%")
        print(f"  mean runtime: {row.mean_runtime:.1f} s")

    out_file = f"outputs/model_comparison_{split}.csv"
    print(f"Saving results to {out_file}")
    df.to_csv(out_file, index=False)

if __name__ == "__main__":
    main()

