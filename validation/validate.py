import sys

import wandb
import torch
from tqdm import tqdm
from torchmetrics import ROC, AUROC
import matplotlib.pyplot as plt
from collections import defaultdict

from common.metrics import get_metrics
from datasets.make_dataset import make_dataloader
from common.old_routine import get_old_model, old_model_key
from common.cfg_utils import get_config, get_run_config
from common.cache import cache
from common.losses import get_losses
from common.plot_metrics import plot_metrics

def pred_key(cfg, run, dataloader, tag, split):
    return (old_model_key(cfg, run, tag), split)

@cache(pred_key, disable=False)
def get_preds(cfg, run, dataloader, tag, split):

    cfg = get_run_config(run, cfg)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = get_old_model(cfg, run, tag).to(device)
    model.eval()
    
    preds = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            pred = model(batch.to(device))
            preds.append(pred)

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

    loss_vals = defaultdict(list)

    print("Getting metrics")
    n_batches = None
    for i, (batch, pred) in enumerate(zip(loader, tqdm(preds))):
        pred = pred.to(device)
        batch = batch.to(device)
        for met in metrics.values():
            met.update(pred, batch)

        loss, loss_dict = get_losses(cfg, batch, pred)
        for name, loss in loss_dict.items():
            loss_vals[name].append(loss)

        if n_batches is not None and i == n_batches:
            break

    ret = { name: met.compute() for name, met in metrics.items() }
    for name, loss in loss_vals.items():
        ret[name] = torch.stack(loss).mean()

    return ret

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
    return metrics

if __name__ == "__main__":
    # run_id, tag, and data_split are all command line args
    # todo: this is a pretty hacky way of getting command line args
    cfg = get_config()
    # validate(cfg, cfg.run_id, cfg.tag, cfg.data_split)
    for name, run_id in (("Lig only", "21mnmh68"), ("Both", "34ednh2q")):
        for split in ("test", "train"):
            print(name, split)
            validate(cfg, run_id, "v4", split)