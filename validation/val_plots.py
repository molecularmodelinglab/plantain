
import matplotlib.pyplot as plt
from collections import defaultdict
from common.utils import flatten_dict
from data_formats.tasks import RejectOption, ScoreActivityClass, ClassifyActivity

def reject_frac_plot(cfg, x, y, pred, metrics):

    fracs = []
    frac_metrics = defaultdict(list)
    for key, val in metrics["select"].items():
        frac = float(key)
        fracs.append(frac)
        for name, metric in flatten_dict(val).items():
            frac_metrics[name].append(metric.item())

    fig, axs = plt.subplots(2, 2)
    graph_data = [
        ("Median AUC", "auroc_median", 0, 0),
        ("Mean AUC", "auroc_mean", 0, 1),
        # ("Accuracy", "acc", 1, 0),
        ("Total AUC", "auroc_all", 1, 0),
        ("Total Accuracy", "acc", 1, 1)
        # ("Total Precision", "precision", 1, 1),
    ]
    for label, metric, x, y in graph_data:
        if metric not in frac_metrics: continue
        ax = axs[x][y]
        ax.plot(fracs, frac_metrics[metric])
        ax.set_xlabel("Fraction selected")
        ax.set_ylabel(label)
    fig.suptitle("Impact of Rejection")
    fig.tight_layout()

    return fig

def act_select_scatter(cfg, x, y, pred, metrics):
    fig, ax = plt.subplots()
    ax.scatter(pred.select_score[y.is_active], pred.active_prob_unnorm[y.is_active], alpha=0.5, label="Actives")
    ax.scatter(pred.select_score[~y.is_active], pred.active_prob_unnorm[~y.is_active], alpha=0.5, label="Inactives")
    ax.set_xlabel("Select score")
    ax.set_ylabel("Activity score")
    ax.legend()

    return fig

def make_plots(cfg, tasks, x, y, pred, metrics):
    plot_funcs = {
        reject_frac_plot: (RejectOption, ScoreActivityClass, ClassifyActivity),
        act_select_scatter: (RejectOption, ScoreActivityClass),
    }
    plots = {}
    for func, plot_tasks in plot_funcs.items():
        if set(plot_tasks).issubset(tasks):
            plots[func.__name__] = func(cfg, x, y, pred, metrics)
    return plots
