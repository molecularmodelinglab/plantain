from typing import Set, Type
import torch
from torch import nn
from torchmetrics import Metric, Accuracy, AUROC, ROC, Precision, R2Score, MeanSquaredError

from data_formats.tasks import Task

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

def get_activity_score(p):
    return p.activity_score

def get_activity(b):
    return b.activity

def identity(b):
    return b

def get_single_task_metrics(task: Type[Task]):
    return {
        "score_activity": nn.ModuleDict({
            "auroc": MetricWrapper(AUROC(), get_activity_score, get_is_active),
        }),
    }[task.get_name()]

def get_metrics(tasks: Set[Type[Task]]):
    ret = nn.ModuleDict({})
    for task in tasks:
        ret.update(get_single_task_metrics(task))
    return ret