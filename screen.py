import sys
sys.path.insert(0, './terrace')

import random
import wandb
import torch
from tqdm import tqdm
from torchmetrics import ROC, AUROC
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from terrace.batch import DataLoader

from models.val_model import OldModel, VinaModel
from common.metrics import get_metrics
from datasets.bigbind_screen import BigBindScreenDataset
from datasets.make_dataset import seed_worker
from common.old_routine import get_old_model, old_model_key, get_weight_artifact
from common.cfg_utils import get_config, get_run_config
from common.cache import cache
from common.plot_metrics import plot_metrics

def pred_key(cfg, model, target, split):
    return (model.get_cache_key(), split, target)

def get_screen_dataloader(cfg, target, split):
    dataset = BigBindScreenDataset(cfg, target, split)
    n_workers = cfg.platform.num_workers
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size,
                            num_workers=n_workers, pin_memory=True,
                            shuffle=False, worker_init_fn=seed_worker)
    return dataloader

@cache(pred_key, disable=False)
def get_screen_preds(cfg, model, target, split):

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    dataloader = get_screen_dataloader(cfg, target, split)
    
    preds = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            preds.append(model(batch.to(device), dataloader.dataset))

    return preds

def get_screen_metrics(scores, yt):
    one_percent = round(len(yt)*0.01)
    if one_percent == 0: return None
    all_pred_and_act = list(zip(scores, yt))
    random.shuffle(all_pred_and_act)
    pred_and_act = sorted(all_pred_and_act, key=lambda x: -x[0])[:one_percent]
    are_active = [ item[1] for item in pred_and_act ]
    tot_actives = sum(yt)
    max_actives = min(tot_actives, one_percent)
    frac_act_chosen = sum(are_active)/len(are_active)
    max_act_frac = max_actives/len(are_active)
    frac_act_in_set = tot_actives/len(yt)
    ef1 = frac_act_chosen/frac_act_in_set
    max_ef1 = max_act_frac/frac_act_in_set
    nef1 = ef1/max_ef1
    return {
        "EF1%": ef1,
        "NEF1%": nef1,
        "total in set": len(yt),
        "total chosen": one_percent,
        "total actives chosen": sum(are_active),
        "total actives in set": tot_actives,
    }

def metrics_key(cfg, model, target, split):
    return (model.get_cache_key(), split, target)

@cache(metrics_key, disable=True)
def get_screen_metric_values(cfg, model, target, split):

    print("Getting predictions")
    preds = get_screen_preds(cfg, model, target, split)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    metrics = get_metrics(cfg)
    for name, met in metrics.items():
        metrics[name] = met.to(device)

    loader = get_screen_dataloader(cfg, target, split)
    if len(loader) < 2:
        return None

    print("Getting metrics")
    n_batches = None
    for i, (batch, pred) in enumerate(zip(loader, tqdm(preds))):
        pred = pred.to(device)
        batch = batch.to(device)
        for met in metrics.values():
            met.update(pred, batch)

        if n_batches is not None and i == n_batches:
            break

    mets = { name: met.compute() for name, met in metrics.items() }

    scores = torch.cat(preds).to('cpu')
    yt = loader.dataset.get_all_yt()

    screen_mets = get_screen_metrics(scores, yt)
    mets.update(screen_mets)

    return mets

def log_metrics(metrics, target):
    for name, val in metrics.items():
        if isinstance(val, torch.Tensor):
            val = val.cpu().numpy()
        elif isinstance(val, (int, float)):
            pass
        else:
            continue
        print(f"{target}_{name}: {val}")

def screen_cache(cfg, val_model, split):
    return (val_model.get_cache_key(), split)

@cache(screen_cache)
def screen(cfg, val_model, split="val"):

    rows = []
    all_targets = BigBindScreenDataset.get_all_targets(cfg, split)
    for target in all_targets:
        print(f"Screening on {target}")
        metrics = get_screen_metric_values(cfg, model, target, split)
        if metrics is None: continue
        log_metrics(metrics, target)
        row = {}
        row["target"] = target
        for name, val in metrics.items():
            if isinstance(val, torch.Tensor):
                val = val.cpu().numpy()
            elif isinstance(val, (int, float)):
                pass
            else:
                continue
            row[name] = val
        rows.append(row)

    df = pd.DataFrame(rows)
    out_filename = f"outputs/{split}_screen_{model.get_name()}.csv"
    print(f"Saving result to {out_filename}")
    df.to_csv(out_filename, index=False)
    return df

if __name__ == "__main__":
    cfg = get_config()
    run_id = "1socj7qg"
    # run_id = "1nhqz8vw"
    use_vina = False
    if use_vina:
        model = VinaModel(cfg)
    else:
        api = wandb.Api()
        run = api.run(f"{cfg.project}/{run_id}")
        cfg = get_run_config(run, cfg)
        model = OldModel(cfg, run, "latest")
    screen(cfg, model)
