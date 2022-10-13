import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.cfg_utils import get_config

def make_screen_fig(cfg, name2df, sort_key="e2ebind"):
    sorted_df = name2df[sort_key]
    sorted_df = sorted_df.sort_values(by="EF1%", ascending=False)

    fig, ax = plt.subplots()
    x = np.arange(len(sorted_df))
    width = 0.25

    name2ef1 = {}
    for i, (name, df) in enumerate(name2df.items()):
        ef1_scores = df["EF1%"][sorted_df.index]
        name2ef1[name] = ef1_scores
        rects = ax.bar(x + width*(i-1)/2, ef1_scores, width, label=name)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('EF1%')
    ax.set_title("Performance on BigBind Screen")
    ax.set_xticks(x)
    targets = [ t.split("_")[0] for t in sorted_df.index ]
    ax.set_xticklabels(targets, rotation='vertical')
    ax.set_xlabel("Target")
    ax.legend()

    fig.set_size_inches(15, 5)
    fig.tight_layout()
    out_filename = "./outputs/val_screen_results.png"
    print(f"Saving figure to {out_filename}")
    fig.savefig(out_filename, transparent=False)

def make_screen_table(cfg, name2df):
    rows = []
    for name, df in name2df.items():
        rows.append({
            "model": name,
            "mean EF1%": df["EF1%"].mean(),
            "median EF1%": df["EF1%"].median(),
            "mean NEF1%": df["NEF1%"].mean(),
            "median NEF1%": df["NEF1%"].median(),
            "mean AUC": df["auroc"].mean(),
            "median AUC": df["auroc"].median(),
        })
    out_df = pd.DataFrame(rows)
    out_filename = "./outputs/val_screen_results.csv"
    print(out_df)
    print(f"Saving table to {out_filename}")
    out_df.to_csv(out_filename, index=False)

if __name__ == "__main__":
    cfg = get_config()
    name2csv = {
        "e2ebind": "./outputs/val_screen_1nhqz8vw_v0.csv",
        "vina": "./outputs/val_screen_vina.csv",
    }
    min_actives = 10
    name2df = {}
    for name, csv in name2csv.items():
        df = pd.read_csv(csv).set_index("target")
        df = df.query("`total actives in set` >= @min_actives")
        name2df[name] = df
    # make_screen_fig(cfg, name2df)
    make_screen_table(cfg, name2df)