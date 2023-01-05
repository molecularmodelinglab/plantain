
from tqdm import tqdm
from datasets.make_dataset import make_dataloader
from validation.metrics import get_metrics


def validate(cfg, model, split, num_batches=None):

    loader = make_dataloader(cfg, split, model.get_data_format())
    tasks = model.get_tasks().intersection(loader.dataset.get_tasks())
    metrics = get_metrics(tasks)

    for i, (x, y) in enumerate(tqdm(loader)):
        pred = model.predict(tasks, x)
        for metric in metrics.values():
            metric.update(pred, y)
        if num_batches is not None and i >= num_batches:
            break

    return {
        key: val.compute() for key, val in metrics.items()
    }