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

@cache(pred_key, disable=False)
def get_preds(cfg, run, dataloader, tag, split):

    cfg = get_run_config(run, cfg)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = get_old_model(cfg, run, tag).to(device)
    
    preds = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            preds.append(model(batch.to(device)))

    return preds

def metrics_key(cfg, run, tag, split):
    return (old_model_key(cfg, run, tag), split)

@cache(metrics_key, disable=False)
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

def plot_many_rocs(rocs, title, out_filename):
    import matplotlib.pyplot as plt
    for name, roc in rocs.items():
        fpr, tpr, thresh = roc
        plt.plot(fpr.cpu(), tpr.cpu(), label=name)
    plt.plot([0, 1], [0, 1], color='black')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.axis('equal')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(title)
    plt.legend()
    plt.savefig(out_filename)
    plt.clf()
    # plt.show()


def make_roc_figs(cfg, tag, split):
    run_ids = {
        "Ligand and receptor": "37jstv82",
        "Ligand only": "exp293if",
    }
    rocs = {}
    for name, run_id in run_ids.items():
        rocs[name] = validate(cfg, run_id, tag, split)["roc"]
    plot_many_rocs(rocs, "With SNA", "outputs/sna_roc.png")
    run_ids = {
        "Ligand and receptor": "1es4be17",
        "Ligand only": "1qwd5qn6",
    }
    rocs = {}
    for name, run_id in run_ids.items():
        rocs[name] = validate(cfg, run_id, tag, split)["roc"]
    plot_many_rocs(rocs, "Without SNA", "outputs/no_sna_roc.png")

if __name__ == "__main__":
    cfg = get_config()
    make_roc_figs(cfg, "v4", "test")
