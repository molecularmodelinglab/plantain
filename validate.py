import sys

import wandb
import torch
from tqdm import tqdm
from torchmetrics import ROC, AUROC
import matplotlib.pyplot as plt

from common.metrics import get_metrics
from datasets.make_dataset import make_dataloader
from common.old_routine import get_old_model, old_model_key
from common.cfg_utils import get_config, get_run_config
from common.cache import cache
from common.plot_metrics import plot_metrics

def pred_key(cfg, run, dataloader, tag, split):
    return (old_model_key(cfg, run, tag), split)

@cache(pred_key, disable=True)
def get_preds(cfg, run, dataloader, tag, split):

    cfg = get_run_config(run, cfg)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = get_old_model(cfg, run, tag).to(device)
    model.eval()
    
    preds = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            preds.append(model(batch.to(device)))

    return preds

def metrics_key(cfg, run, tag, split):
    return (old_model_key(cfg, run, tag), split)

@cache(metrics_key, disable=True)
def get_metric_values(cfg, run, tag, split):

    cfg = get_run_config(run, cfg)

    loader = make_dataloader(cfg, split)

    print("Getting predictions")
    preds = get_preds(cfg, run, loader, tag, split)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    metrics = get_metrics(cfg)
    for name, met in metrics.items():
        metrics[name] = met.to(device)

    print("Getting metrics")
    n_batches = None
    for i, (batch, pred) in enumerate(zip(loader, tqdm(preds))):
        pred = pred.to(device)
        batch = batch.to(device)
        for met in metrics.values():
            met.update(pred, batch)

        if n_batches is not None and i == n_batches:
            break

    return { name: met.compute() for name, met in metrics.items() }

def log_metrics(run, metrics, split):
    for name, val in metrics.items():
        if not isinstance(val, torch.Tensor): continue
        print(f"{split}_{name}: {val}")

def validate(cfg, run_id, tag, split, to_wandb=False):

    if to_wandb:
        run = wandb.init(project=cfg.project, id=run_id, resume=True)
    else:
        api = wandb.Api()
        run = api.run(f"{cfg.project}/{run_id}")
    cfg = get_run_config(run, cfg)

    metrics = get_metric_values(cfg, run, tag, split)
    log_metrics(run, metrics, split)
    # plot_metrics(metrics, split, to_wandb, run)
    return metrics

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

def validate_regression(cfg):
    print("Validating Ligand and Receptor on test set")
    validate(cfg, "34ednh2q", "v4", "test")
    print("Validating Ligand only on test set")
    validate(cfg, "21mnmh68", "v4", "test")


    print("Validating Ligand and Receptor on train set")
    validate(cfg, "34ednh2q", "v4", "train")
    print("Validating Ligand only on train set")
    validate(cfg, "21mnmh68", "v4", "train")

if __name__ == "__main__":
    cfg = get_config()
    make_roc_figs(cfg, "v4", "test")
    # validate_regression(cfg)