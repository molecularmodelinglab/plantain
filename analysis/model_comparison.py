
import pandas as pd
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

    out_metrics = { "name": name }
    out_metrics["subset"] = "none" if subset is None else subset
    for key, val in flatten_dict(metrics).items():
        out_metrics[key] = float(val)

    if name == "plantain":
        out_metrics["mean_runtime"] = sum(runtimes)/len(runtimes)

    return out_metrics


def main():
    split = "test"
    model_names = [
        "diffdock",
        # "plantain",
        # "gnina",
        # "vina",
    ]
    data = []
    for name in model_names:
        data.append(eval_model_on_crossdocked(name, "test", "diffdock"))

    df = pd.DataFrame(data)

    print("Final results:\n")

    for row in df.itertuples():
        print(f"{row.name} results (subset={row.subset})")
        print(f"  <2 Å acc:")
        print(f"    mean: {row.acc_2_mean_1*100:.0f}%, unnorm: {row.acc_2_all_1*100:.0f}%")
        print(f"  <5 Å acc:")
        print(f"    mean: {row.acc_5_mean_1*100:.0f}%, unnorm: {row.acc_5_all_1*100:.0f}%")
        if "mean_runtime" in row:
            print(f"  mean runtime: {row.mean_runtime:.0f} s")
        # print()

    out_file = f"outputs/model_comparison_{split}.csv"
    print(f"Saving results to {out_file}")
    df.to_csv(out_file, index=False)

if __name__ == "__main__":
    main()

