from terrace import Batch, collate
from typing import Callable, Set, Type
import torch
from torch import nn
from torchmetrics import Metric, SpearmanCorrCoef, Accuracy, AUROC, ROC, Precision, R2Score, MeanSquaredError
from data_formats.base_formats import Input, LigAndRec, Prediction, Label

from data_formats.tasks import Task

class FullMetric(Metric):
    """ Extension of torchmetrics metrics which also allows us
    to update based on input batches """

    def update(self, x: Batch[Input], pred: Batch[Prediction], label: Batch[Label]):
        raise NotImplementedError()

    def compute(self):
        raise NotImplementedError

class MetricWrapper(FullMetric):

    def __init__(self, metric_class: Type[Metric], get_y_p: Callable, get_y_t: Callable):
        super().__init__()
        self.metric = metric_class()
        self.get_y_p = get_y_p
        self.get_y_t = get_y_t

    def update(self, x: Batch[Input], pred: Batch[Prediction], label: Batch[Label]):
        y_pred = self.get_y_p(pred)
        y_true = self.get_y_t(label)
        return self.metric.update(y_pred, y_true)

    def compute(self):
        return self.metric.compute()


class PerPocketMetric(FullMetric):

    def __init__(self, metric_class: Type[FullMetric]):
        """ reduce is either mean or median"""
        super().__init__()
        self.metric_class = metric_class
        self.metric = metric_class()
        self.pocket_metrics = nn.ModuleDict()

    def update(self, x_b: Batch[LigAndRec], pred_b: Batch[Prediction], label_b: Batch[Label]):
        self.metric.update(x_b, pred_b, label_b)
        for x, pred, label in zip(x_b, pred_b, label_b):
            poc_id = x.pocket_id
            xi = collate([x])
            predi = collate([pred])
            labeli = collate([label])
            if poc_id not in self.pocket_metrics:
                self.pocket_metrics[poc_id] = self.metric_class()
            self.pocket_metrics[poc_id].update(xi, predi, label)

    def pre_compute(self):
        pass

    def compute(self):
        self.pre_compute()
        results = []
        for metric in self.pocket_metrics.values():
            results.append(metric.compute())
        results = torch.stack(results)
        return {
            "all": self.metric.compute(),
            "mean": results.mean(),
            "median": results.median()
        }

class PerPocketAUC(PerPocketMetric):

    def __init__(self):
        super().__init__(IsActiveAUC)

    def pre_compute(self):
        """ Can't compute AUC for labels that are all active or all inactive"""
        for key, val in list(self.pocket_metrics.items()):
            if True not in val.metric.target or False not in val.metric.target:
                del self.pocket_metrics[key]

# needed because the pickle can't handle lambdas
def get_is_active(b):
    return b.is_active

def get_is_active_score(p):
    return p.is_active_score

def get_activity_score(p):
    return p.activity_score

def get_activity(b):
    return b.activity

def identity(b):
    return b

class IsActiveAUC(MetricWrapper):

    def __init__(self):
        super().__init__(AUROC, get_is_active_score, get_is_active)

class ActivitySpearman(MetricWrapper):

    def __init__(self):
        super().__init__(SpearmanCorrCoef, get_activity_score, get_activity)

def get_single_task_metrics(task: Type[Task]):
    return {
        "score_activity_class": nn.ModuleDict({
            "auroc": PerPocketAUC()
        }),
        "score_activity_regr": nn.ModuleDict({
            "spearman": PerPocketMetric(ActivitySpearman),
        })
    }[task.get_name()]

def get_metrics(tasks: Set[Type[Task]]):
    ret = nn.ModuleDict({})
    for task in tasks:
        ret.update(get_single_task_metrics(task))
    return ret