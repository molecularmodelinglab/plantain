import sys

import random
import wandb
import torch
from tqdm import tqdm
from torchmetrics import ROC, AUROC
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from terrace.batch import DataLoader

from models.val_model import OldModel, VinaModel, GninaModel, ComboModel
from common.metrics import get_metrics
from datasets.bigbind_screen import BigBindScreenDataset
from datasets.lit_pcba import LitPcbaDataset
from datasets.make_dataset import seed_worker
from common.old_routine import get_old_model, old_model_key, get_weight_artifact
from common.cfg_utils import get_config, get_run_config
from common.cache import cache
from common.plot_metrics import plot_metrics

def get_bigbind_screen_dataloader(cfg, target, split):
    dataset = BigBindScreenDataset(cfg, target, split)
    n_workers = cfg.platform.num_workers
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size,
                            num_workers=n_workers, pin_memory=True,
                            shuffle=False, worker_init_fn=seed_worker)
    return dataloader

def get_lit_pcba_dataloader(cfg, target, split):
    dataset = LitPcbaDataset(cfg, target)
    n_workers = cfg.platform.num_workers
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size,
                            num_workers=n_workers, pin_memory=True,
                            shuffle=False, worker_init_fn=seed_worker)
    return dataloader

def pred_key(cfg, model, dataset_name, target, split):
    return (model.get_cache_key(), dataset_name, target, split)

def get_screen_dataloader(cfg, dataset_name, target, split):
    return {
        "lit_pcba": get_lit_pcba_dataloader,
        "bigbind": get_bigbind_screen_dataloader,
    }[dataset_name](cfg, target, split)

@cache(pred_key, disable=False)
def get_screen_preds(cfg, model, dataset_name, target, split):

    if isinstance(model, ComboModel):
        return [ get_screen_preds(cfg, submodel, dataset_name, target, split) for submodel in (model.model1, model.model2) ]

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    dataloader = get_screen_dataloader(cfg, dataset_name, target, split)
    
    preds = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            preds.append(model(batch.to(device), dataloader.dataset))

    return preds

def get_screen_metrics(scores, model, yt):
    one_percent = round(len(yt)*0.01)
    if one_percent == 0: return None
    if isinstance(model, ComboModel):
        are_active = [ yt[idx] for idx in model.choose_topk(one_percent)]
    else:
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

def recur_to(item, device):
    try:
        return item.to(device)
    except AttributeError:
        assert isinstance(item, list)
        return [ recur_to(i, device) for i in item ]
    
@cache(pred_key, disable=False)
def get_screen_metric_values(cfg, model, dataset_name, target, split):

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    print(f"Getting predictions for {model.get_name()} on {dataset_name} {target}")
    preds = get_screen_preds(cfg, model, dataset_name, target, split)
    # preds = recur_to(preds, device)

    if isinstance(model, ComboModel):
        # can't do AUROC etc on combo model
        model.init_preds(*[ torch.cat(p).to('cpu') for p in preds ])
        metrics = {}

    else:
        metrics = get_metrics(cfg)
        for name, met in metrics.items():
            metrics[name] = met.to(device)

    loader = get_screen_dataloader(cfg, dataset_name, target, split)
    if len(loader) < 2:
        return None

    print("Getting metrics")
    if len(metrics) > 0:
        n_batches = None
        for i, (batch, pred) in enumerate(zip(loader, tqdm(preds))):
            pred = pred.to(device)
            batch = batch.to(device)
            for met in metrics.values():
                met.update(pred, batch)

            if n_batches is not None and i == n_batches:
                break

    mets = { name: met.compute() for name, met in metrics.items() }

    if isinstance(model, ComboModel):
        scores = None
    else:
        scores = torch.cat(preds).to('cpu')
    yt = loader.dataset.get_all_yt()

    screen_mets = get_screen_metrics(scores, model, yt)
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

def screen_key(cfg, model, dataset_name, split):
    return (model.get_cache_key(), dataset_name, split)

@cache(screen_key, disable=True)
def screen(cfg, model, dataset_name, split):

    all_targets = {
        "bigbind": BigBindScreenDataset.get_all_targets(cfg, split),
        "lit_pcba": LitPcbaDataset.get_all_targets(cfg),
    }[dataset_name]

    rows = []
    for target in all_targets:
        # todo: only for now
        # if target == "IDH1": continue
        print(f"Screening on {target}")
        metrics = get_screen_metric_values(cfg, model, dataset_name, target, split)
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
    out_filename = f"outputs/screen_{dataset_name}_{split}_{model.get_name()}.csv"
    print(f"Saving result to {out_filename}")
    df.to_csv(out_filename, index=False)
    return df

def get_run_val_model(cfg, run_id, tag):
    api = wandb.Api()
    run = api.run(f"{cfg.project}/{run_id}")
    cfg = get_run_config(run, cfg)
    model = OldModel(cfg, run, tag)
    model.model.eval()
    return model, cfg

if __name__ == "__main__":
    cfg = get_config()
    gnina = GninaModel(cfg)
    e2ebind, cfg = get_run_val_model(cfg, "37jstv82", "v4")
    combo = ComboModel(e2ebind, gnina, 0.1)
    screen(cfg, e2ebind, "bigbind", "test")
    screen(cfg, combo, "lit_pcba", "test")
    screen(cfg, e2ebind, "lit_pcba", "test")
