
import matplotlib.pyplot as plt

from common.cfg_utils import get_config
from validation.validate import validate

def plot_many_rocs(ax, rocs, title):
    for name, roc in rocs.items():
        fpr, tpr, thresh = roc
        ax.plot(fpr.cpu(), tpr.cpu(), label=name)
    ax.plot([0, 1], [0, 1], color='black')
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.axis('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title)
    ax.legend()

def make_roc_figs(cfg, tag, split):
    fig, (ax1, ax2) = plt.subplots(1, 2)

    run_ids = {
        "Ligand and receptor": "37jstv82",
        "Ligand only": "exp293if",
    }
    rocs = {}
    for name, run_id in run_ids.items():
        print(f"Validating SNA {name}")
        rocs[name] = validate(cfg, run_id, tag, split)["roc"]
    plot_many_rocs(ax1, rocs, "With SNA")

    run_ids = {
        "Ligand and receptor": "1es4be17",
        "Ligand only": "1qwd5qn6",
    }
    rocs = {}
    for name, run_id in run_ids.items():
        print(f"Validating Non-SNA {name}")
        rocs[name] = validate(cfg, run_id, tag, split)["roc"]
    plot_many_rocs(ax2, rocs, "Without SNA")

    fig.tight_layout()
    fig.set_size_inches(6, 3.5)
    fig.savefig("./outputs/roc.png", dpi=300)

if __name__ == "__main__":
    cfg = get_config()
    make_roc_figs(cfg, "v4", "test")