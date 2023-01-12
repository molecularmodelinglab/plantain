import torch
from tqdm import tqdm
from datasets.make_dataset import make_dataloader
from validation.metrics import get_metrics
from common.utils import flatten_dict

@torch.no_grad()
def validate(cfg, model, dataset_name, split, num_batches=None):

    loader = make_dataloader(cfg, dataset_name, split, model.get_data_format())
    tasks = model.get_tasks().intersection(loader.dataset.get_tasks())
    metrics = get_metrics(tasks)

    for i, (x, y) in enumerate(tqdm(loader)):
        pred = model.predict(tasks, x)
        for metric in metrics.values():
            metric.update(x, pred, y)
        if num_batches is not None and i >= num_batches:
            break

    ret = {
        key: val.compute() for key, val in metrics.items()
    }
    return flatten_dict(ret)