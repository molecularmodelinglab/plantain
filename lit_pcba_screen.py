import sys
sys.path.insert(0, './terrace')

import wandb
import torch
from tqdm import tqdm
from torchmetrics import ROC, AUROC
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from terrace.batch import DataLoader

from common.metrics import get_metrics
from datasets.lit_pcba import LitPcbaDataset
from datasets.make_dataset import seed_worker
from common.old_routine import get_old_model, old_model_key, get_weight_artifact
from common.cfg_utils import get_config, get_run_config
from common.cache import cache
from common.plot_metrics import plot_metrics

def pred_key(cfg, run, target, tag):
    return (old_model_key(cfg, run, tag), target)

def get_lit_pcba_dataloader(cfg, target):
    dataset = LitPcbaDataset(cfg, target)
    n_workers = cfg.platform.num_workers
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size,
                            num_workers=n_workers, pin_memory=True,
                            shuffle=False, worker_init_fn=seed_worker)
    return dataloader

@cache(pred_key, disable=False)
def get_lit_pcba_preds(cfg, run, target, tag="latest"):

    cfg = get_run_config(run, cfg)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = get_old_model(cfg, run, tag).to(device)

    dataloader = get_lit_pcba_dataloader(cfg, target)
    
    preds = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            preds.append(model(batch.to(device)))

    return preds

def get_screen_metrics(scores, yt):
    one_percent = round(len(yt)*0.01)
    if one_percent == 0: return None
    pred_and_act = sorted(zip(scores, yt), key=lambda x: -x[0])[:one_percent]
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

def metrics_key(cfg, run, target, tag):
    return (old_model_key(cfg, run, tag), target)

@cache(metrics_key, disable=False)
def get_lit_pcba_metric_values(cfg, run, target, tag="latest"):

    cfg = get_run_config(run, cfg)

    print("Getting predictions")
    preds = get_lit_pcba_preds(cfg, run, target, tag)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    metrics = get_metrics(cfg)
    for name, met in metrics.items():
        metrics[name] = met.to(device)

    loader = get_lit_pcba_dataloader(cfg, target)
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

    target_mets = get_screen_metrics(scores, yt)
    if target_mets is not None:
        mets.update(target_mets)

    return mets

def log_metrics(run, metrics, target):
    for name, val in metrics.items():
        if isinstance(val, torch.Tensor):
            val = val.cpu().numpy()
        elif isinstance(val, (int, float)):
            pass
        else:
            continue
        print(f"{target}_{name}: {val}")

def make_lit_pcba_fig(cfg, run, df):

    prior_lit_pcba = pd.read_csv("prior_work/lit_pcba.csv")
    prior_lit_pcba = prior_lit_pcba.set_index("target")

    df = df.sort_values(by="EF1%", ascending=False)

    fig, ax = plt.subplots()
    x = np.arange(len(df))
    width = 0.25

    ef1_scores = df["EF1%"]
    gnina_ef1 = prior_lit_pcba["gnina EF1%"][df["target"]]
    vina_ef1 = prior_lit_pcba["vina EF1%"][df["target"]]

    rects1 = ax.bar(x - width, ef1_scores, width, label='e2ebind')
    rects2 = ax.bar(x, gnina_ef1, width, label='gnina')
    rects3 = ax.bar(x + width, vina_ef1, width, label='vina')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('EF1%')
    ax.set_title('Performance on LIT-PCBA')
    ax.set_xticks(x)
    ax.set_xticklabels(list(df["target"]), rotation='vertical')
    ax.set_xlabel("Target")
    ax.set_aspect(0.5)
    ax.legend()

    fig.tight_layout()
    artifact = get_weight_artifact(run)
    out_filename = f"outputs/lit_pca_results_{run_id}_{artifact.version}.png"
    fig.savefig(out_filename, transparent=False)

@cache(old_model_key)
def lit_pcba_screen(cfg, run, tag="latest"):

    rows = []
    all_targets = LitPcbaDataset.get_all_targets(cfg)
    for target in all_targets:
        print(f"Screening on {target}")
        metrics = get_lit_pcba_metric_values(cfg, run, target, tag)
        log_metrics(run, metrics, target)
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
    artifact = get_weight_artifact(run)
    out_filename = f"outputs/lit_pca_results_{run_id}_{artifact.version}.csv"
    print(f"Saving result to {out_filename}")
    df.to_csv(out_filename, index=False)
    print(f"Making figure")
    make_lit_pcba_fig(cfg, run, df)
    return df

if __name__ == "__main__":
    cfg = get_config()
    # run_id = "1socj7qg"
    run_id = "1nhqz8vw"
    api = wandb.Api()
    run = api.run(f"{cfg.project}/{run_id}")
    cfg = get_run_config(run, cfg)
    lit_pcba_screen(cfg, run)
