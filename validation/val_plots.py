
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KernelDensity
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

def act_select_scatter(cfg, x, y, pred, metrics, select_score=None):
    if select_score is None:
        select_score = pred.select_score
    fig, ax = plt.subplots()
    ax.scatter(select_score[y.is_active], pred.active_prob_unnorm[y.is_active], alpha=0.5, label="Actives")
    ax.scatter(select_score[~y.is_active], pred.active_prob_unnorm[~y.is_active], alpha=0.5, label="Inactives")
    ax.set_xlabel("Select score")
    ax.set_ylabel("Activity score")
    ax.legend()

    return fig

def uncertainty_kde(cfg, x, y, pred, metrics, select_score=None):
    if select_score is None:
        select_score = pred.select_score
    U = -select_score
    S = pred.active_prob
    US = torch.stack((U, S)).T
    act_kde = KernelDensity(bandwidth=0.1, kernel='gaussian').fit(US[y.is_active])
    inact_kde = KernelDensity(bandwidth=0.1, kernel='gaussian').fit(US[~y.is_active])

    U = torch.linspace(0.0, 1.0, 100)
    S = torch.linspace(0.0, 1.0, 100)
    UU, SS = torch.meshgrid(U, S)
    UUSS = torch.stack([UU.reshape(-1), SS.reshape(-1)]).T
    act_dense = np.exp(act_kde.score_samples(UUSS).reshape((100, 100)))
    inact_dense = np.exp(inact_kde.score_samples(UUSS).reshape((100, 100)))

    alpha = 1.0
    prob = (act_dense + 0.4*alpha)/(act_dense + inact_dense + alpha)

    fig, ax = plt.subplots()#subplot_kw={"projection": "3d"})
    cf = ax.contourf(UU, SS, prob, cmap='Blues')
    ax.set_xlabel("Uncertainty")
    ax.set_ylabel("Prediction")
    ax.set_title("P(Active)")
    # ax.plot_surface(UU, SS, P_val)
    fig.colorbar(cf)

    return fig

def uncertainty_gaussian(cfg, x, y, pred, metrics, select_score=None):
    if select_score is None:
        select_score = pred.select_score
    U = -select_score
    S = pred.active_prob
    US = torch.stack((U, S)).T
    gpc = GaussianProcessClassifier().fit(US, y.is_active)

    U = torch.linspace(0.0, 1.0, 100)
    S = torch.linspace(0.0, 1.0, 100)
    UU, SS = torch.meshgrid(U, S)
    UUSS = torch.stack([UU.reshape(-1), SS.reshape(-1)]).T

    prob = gpc.predict_proba(UUSS)[:,1].reshape((100, 100))

    fig, ax = plt.subplots()#subplot_kw={"projection": "3d"})
    cf = ax.contourf(UU, SS, prob, cmap='Blues')
    ax.set_xlabel("Uncertainty")
    ax.set_ylabel("Prediction")
    ax.set_title("P(Active)")
    # ax.plot_surface(UU, SS, P_val)
    fig.colorbar(cf)

    return fig

def ideal_uncertainty_plot():
    SS, UU = np.mgrid[0.0:1.0:100j, 0.0:1.0:100j]
    P_val = (1-UU.ravel())*SS.ravel() + UU.ravel()*0.4
    P_val = P_val.reshape((100,100))
    fig, ax = plt.subplots()#subplot_kw={"projection": "3d"})
    cf = ax.contourf(UU, SS, P_val, cmap='Blues')
    ax.set_xlabel("Uncertainty")
    ax.set_ylabel("Prediction")
    ax.set_title("P(Active) for ideal uncertainty metric")
    # ax.plot_surface(UU, SS, P_val)
    fig.colorbar(cf)
    fig.savefig("outputs/ideal_uncertainty.png")

def make_plots(cfg, tasks, x, y, pred, metrics):
    plot_funcs = {
        reject_frac_plot: (RejectOption, ScoreActivityClass, ClassifyActivity),
        act_select_scatter: (RejectOption, ScoreActivityClass),
        uncertainty_kde: (RejectOption, ClassifyActivity),
        uncertainty_gaussian: (RejectOption, ClassifyActivity)
    }
    plots = {}
    for func, plot_tasks in plot_funcs.items():
        if set(plot_tasks).issubset(tasks):
            plots[func.__name__] = func(cfg, x, y, pred, metrics)
    return plots
