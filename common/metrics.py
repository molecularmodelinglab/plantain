import torch
from torch import nn
from torchmetrics import Metric, Accuracy, AUROC, ROC, Precision, R2Score, MeanSquaredError

from common.cfg_utils import get_all_tasks

class MetricWrapper(Metric):

    def __init__(self, metric, get_y_p, get_y_t):
        super().__init__()
        self.metric = metric
        self.get_y_p = get_y_p
        self.get_y_t = get_y_t

    def update(self, pred, batch):
        y_pred = self.get_y_p(pred)
        y_true = self.get_y_t(batch)
        return self.metric.update(y_pred, y_true)

    def compute(self):
        return self.metric.compute()

# needed because the pickle can't handle lambdas
def get_is_active(b):
    return b.is_active

def get_activity(b):
    return b.activity

def identity(b):
    return b

def get_single_task_metric(task):
    return {
        "classification": nn.ModuleDict({
            "acc": MetricWrapper(Accuracy(), torch.sigmoid, get_is_active),
            "bal_acc": MetricWrapper(Accuracy(average="macro", num_classes=2, multiclass=True), torch.sigmoid, get_is_active),
            "auroc": MetricWrapper(AUROC(), torch.sigmoid, get_is_active),
            "precision": MetricWrapper(Precision(), torch.sigmoid, get_is_active),
            "roc": MetricWrapper(ROC(), torch.sigmoid, get_is_active)
        }),
        "regression": nn.ModuleDict({
            "r2": MetricWrapper(R2Score(), identity, get_activity),
            "mse": MetricWrapper(MeanSquaredError(), identity, get_activity)
        }),
        "pose": nn.ModuleDict({}),
        "pose_rank": nn.ModuleDict({}),
    }[task]

def get_metrics(cfg):
    ret = nn.ModuleDict({})
    for task in get_all_tasks(cfg):
        ret.update(get_single_task_metric(task))
    return ret