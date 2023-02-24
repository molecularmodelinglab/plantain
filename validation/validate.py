import os
import pickle
import resource
from terrace import collate
from terrace.batch import DataLoader
from validation.val_plots import make_plots

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (32768, rlimit[1]))

import torch
from tqdm import tqdm
from common.cache import cache
from datasets.make_dataset import make_dataloader, seed_worker
from validation.metrics import get_metrics
from common.utils import flatten_dict

def pred_key(cfg, model, dataset_name, split, num_batches):
    return (model.cache_key, dataset_name, split, num_batches)

@cache(pred_key)
@torch.no_grad()
def get_preds(cfg, model, dataset_name, split, num_batches):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    loader = make_dataloader(cfg, dataset_name, split, model.get_input_feats())
    if num_batches is not None:
        # shuffle to get better sample
        loader = DataLoader(loader.dataset,
                            batch_size=cfg.batch_size,
                            num_workers=loader.num_workers,
                            pin_memory=True,
                            shuffle=True,
                            worker_init_fn=seed_worker)

    tasks = set(model.get_tasks()).intersection(loader.dataset.get_tasks())

    xs = []
    ys = []
    preds = []

    for i, (x, y) in enumerate(tqdm(loader)):
        x = x.to(device)
        y = y.to(device)
        pred = model.predict(x, tasks)
        if num_batches is not None and i >= num_batches:

            break

        for x0, y0, pred0 in zip(x, y, pred):
            xs.append(x0)
            ys.append(y0)
            preds.append(pred0)

    return collate(xs), collate(ys), collate(preds)

@torch.no_grad()
def validate(cfg, model, dataset_name, split, num_batches=None):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    loader = make_dataloader(cfg, dataset_name, split, model.get_input_feats())
    tasks = set(model.get_tasks()).intersection(loader.dataset.get_tasks())
    metrics = get_metrics(cfg, tasks, offline=True).to(device)

    x, y, pred = get_preds(cfg, model, dataset_name, split, num_batches)
    for metric in metrics.values():
        metric.update(x, pred, y)

    comp_mets = {
        key: val.cpu().compute() for key, val in metrics.items()
    }
    plots = make_plots(cfg, tasks, x.cpu(), y.cpu(), pred.cpu(), comp_mets)
    return comp_mets, plots # flatten_dict(ret)

def save_validation(cfg, model, dataset_name, split, num_batches=None):
    metrics, plots = validate(cfg, model, dataset_name, split, num_batches)
    
    out_folder =f"outputs/results/{model.cache_key}/{dataset_name}_{split}"

    for name, val in flatten_dict(metrics).items():
        if isinstance(val, torch.Tensor):
            val = val.item()
        print(f"  {name}: {val:.2f}")
    print(f"Saving metrics and plots to {out_folder}")

    os.makedirs(out_folder, exist_ok=True)

    metric_fname = out_folder + "/metrics.pkl"
    with open(metric_fname, "wb") as f:
        pickle.dump(metrics, f)

    for name, fig in plots.items():
        fig.savefig(f"{out_folder}/{name}.png")

    return out_folder
