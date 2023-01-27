import resource
from terrace import collate

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

import torch
from tqdm import tqdm
from common.cache import cache
from datasets.make_dataset import make_dataloader
from validation.metrics import get_metrics
from common.utils import flatten_dict

def pred_key(cfg, model, dataset_name, split, num_batches):
    return (model.cache_key, dataset_name, split, num_batches)

# @cache(pred_key)
def get_preds(cfg, model, dataset_name, split, num_batches):
    loader = make_dataloader(cfg, dataset_name, split, model.get_data_format())
    tasks = set(model.get_tasks()).intersection(loader.dataset.get_tasks())

    xs = []
    ys = []
    preds = []

    for i, (x, y) in enumerate(tqdm(loader)):
        pred = model.predict(tasks, x)
        if num_batches is not None and i >= num_batches:

            break

        for x0, y0, pred0 in zip(x, y, pred):
            xs.append(x0)
            ys.append(y0)
            preds.append(pred0)

    return collate(xs), collate(ys), collate(preds)

@torch.no_grad()
def validate(cfg, model, dataset_name, split, num_batches=None):

    loader = make_dataloader(cfg, dataset_name, split, model.get_data_format())
    tasks = set(model.get_tasks()).intersection(loader.dataset.get_tasks())
    metrics = get_metrics(tasks)

    x, y, pred = get_preds(cfg, model, dataset_name, split, num_batches)
    # for metric in metrics.values():
    #     metric.update(x, pred, y)

    ret = {}

    select_score = -pred.pred_bce
    idx = torch.argsort(select_score, dim=0)
    prev_idx = 0
    for frac in torch.arange(0.05, 1.05, 0.05):
        length = int(len(idx)*frac)
        if length == prev_idx: continue
        for index in idx[prev_idx:length]:
            x0 = collate([x[index]])
            y0 = collate([y[index]])
            pred0 = collate([pred[index]])
            for metric in metrics.values():
                metric.update(x0, pred0, y0)
        prev_idx = length

        ret[f"frac_{frac:.2f}"] = {
            key: val.compute() for key, val in metrics.items()
        }

    # for i, (x, y) in enumerate(tqdm(loader)):
    #     pred = model.predict(tasks, x)
    #     for metric in metrics.values():
    #         metric.update(x, pred, y)
    #     if num_batches is not None and i >= num_batches:
    #         break

    # ret = {
    #     key: val.compute() for key, val in metrics.items()
    # }
    return ret # flatten_dict(ret)