from collections import defaultdict
from copy import deepcopy
from terrace import Batch, collate
from typing import Callable, Dict, Set, Type
import torch
from torch import nn
from torchmetrics import Metric, SpearmanCorrCoef, Accuracy, AUROC, ROC, Precision, R2Score, MeanSquaredError
from data_formats.base_formats import Input, LigAndRec, Prediction, Label

from data_formats.tasks import RejectOption, Task

class FullMetric(Metric):
    """ Extension of torchmetrics metrics which also allows us
    to update based on input batches """

    def update(self, x: Batch[Input], pred: Batch[Prediction], label: Batch[Label]):
        raise NotImplementedError()

    def compute(self):
        raise NotImplementedError

class MetricWrapper(FullMetric):

    def __init__(self, metric_class: Type[Metric], get_y_p: Callable, get_y_t: Callable, *args, **kwargs):
        super().__init__()
        self.metric = metric_class(*args, **kwargs)
        self.get_y_p = get_y_p
        self.get_y_t = get_y_t

    def update(self, x: Batch[Input], pred: Batch[Prediction], label: Batch[Label]):
        y_pred = self.get_y_p(pred)
        y_true = self.get_y_t(label)
        return self.metric.update(y_pred, y_true)

    def compute(self):
        return self.metric.compute()


class PerPocketMetric(FullMetric):
    """ Computes a specific metric for every pocket and returns the 
    mean and median values """

    def __init__(self, metric_class: Type[FullMetric], *args, **kwargs):
        """ reduce is either mean or median"""
        super().__init__()
        self.metric_maker = lambda: metric_class(*args, **kwargs)
        self.metric = self.metric_maker()
        self.pocket_metrics = nn.ModuleDict()

    def update(self, x_b: Batch[LigAndRec], pred_b: Batch[Prediction], label_b: Batch[Label]):
        self.metric.update(x_b, pred_b, label_b)
        for x, pred, label in zip(x_b, pred_b, label_b):
            poc_id = x.pocket_id
            xi = collate([x])
            predi = collate([pred])
            labeli = collate([label])
            if poc_id not in self.pocket_metrics:
                self.pocket_metrics[poc_id] = self.metric_maker()
            self.pocket_metrics[poc_id].update(xi, predi, labeli)

    def pre_compute(self):
        pass

    def compute(self):
        self.pre_compute()
        results = []
        for metric in self.pocket_metrics.values():
            results.append(metric.compute())
        if len(results) == 0:
            return {
                "all": self.metric.compute()
            }
        else:
            results = torch.stack(results)
            return {
                "all": self.metric.compute(),
                "mean": results.mean(),
                "median": results.median()
            }

# needed because the pickle can't handle lambdas
def get_is_active(b):
    return b.is_active

def get_is_active_score(p):
    return p.is_active_score

def get_activity_score(p):
    return p.activity_score

def get_active_prob(p):
    return p.active_prob

def get_activity(b):
    return b.activity

def identity(b):
    return b

class IsActiveAUC(MetricWrapper):

    def __init__(self):
        super().__init__(AUROC, get_is_active_score, get_is_active, 'binary')

    def reset(self):
        super().reset()

class PerPocketAUC(PerPocketMetric):

    def __init__(self):
        super().__init__(IsActiveAUC)

    def pre_compute(self):
        """ Can't compute AUC for labels that are all active or all inactive"""
        for key, val in list(self.pocket_metrics.items()):
            if True not in val.metric.target or False not in val.metric.target:
                del self.pocket_metrics[key]

class ActivitySpearman(MetricWrapper):

    def __init__(self):
        super().__init__(SpearmanCorrCoef, get_activity_score, get_activity)

class PoseAcc(FullMetric):

    def __init__(self, rmsd_cutoff=2.0):
        super().__init__()
        self.rmsd_cutoff = rmsd_cutoff
        self.top_n_correct = defaultdict(int)
        self.total_seen = 0

    def update(self, x, pred, y):
        for pred0, y0 in zip(pred, y):
            # if torch.amax(pred0.pose_scores) < -0.25:
            #     print("skipping")
            #     continue
            # print("doing!")
            poses_correct = y0.pose_rmsds < self.rmsd_cutoff
            correct_sorted = [ x[0].cpu().item() for x in sorted(zip(poses_correct, -pred0.pose_scores, torch.randn_like(pred0.pose_scores)), key=lambda x: x[1:])]
            for n in range(1, len(correct_sorted)+1):
                self.top_n_correct[n] += int(True in correct_sorted[:n])
            self.total_seen += 1

    def compute(self):
        # print("TOTAL: ", self.total_seen)
        ret = {}
        for n, correct in self.top_n_correct.items():
            ret[f"top_{n}"] = correct/self.total_seen
        return ret
        
class RejectOptionMetric(FullMetric):

    def __init__(self, cfg, metrics):
        super().__init__()
        self.select_fracs = torch.arange(0.05, 1.05, 0.05)
        self.metrics = nn.ModuleDict(metrics)
        self.xs = []
        self.ys = []
        self.preds = []

    def update(self, x, pred, y):
        for x0, pred0, y0 in zip(x, pred, y):
            self.xs.append(x0)
            self.ys.append(y0)
            self.preds.append(pred0)

    def reset(self):
        super().reset()
        self.xs = []
        self.ys = []
        self.preds = []

    def compute(self):
        ret = {}
        x, y, pred = collate(self.xs), collate(self.ys), collate(self.preds)
        select_score = -pred.select_score
        idx = torch.argsort(select_score, dim=0)
        prev_idx = 0
        for frac in self.select_fracs:
            length = int(len(idx)*frac)
            if length == prev_idx: continue
            for index in idx[prev_idx:length]:
                x0 = collate([x[index]])
                y0 = collate([y[index]])
                pred0 = collate([pred[index]])
                for metric in self.metrics.values():
                    metric.update(x0, pred0, y0)
            prev_idx = length

            ret[f"{frac:.2f}"] = {
                key: val.compute() for key, val in self.metrics.items()
            }
        return ret

def get_single_task_metrics(task: Type[Task]):
    return {
        "score_activity_class": nn.ModuleDict({
            "auroc": PerPocketAUC()
        }),
        "score_activity_regr": nn.ModuleDict({
            "spearman": PerPocketMetric(ActivitySpearman),
        }),
        "classify_activity": nn.ModuleDict({
            "acc": MetricWrapper(Accuracy, get_active_prob, get_is_active, 'binary'),
            "precision": MetricWrapper(Precision, get_active_prob, get_is_active, 'binary')
        }),
        "score_pose": nn.ModuleDict({
            "acc_2": PoseAcc(2.0),
            "acc_5": PoseAcc(5.0)
        }),
        "predict_interaction_mat": nn.ModuleDict(),
    }[task.get_name()]

def get_metrics(cfg, tasks: Set[Type[Task]]):
    ret = nn.ModuleDict({})
    for task in tasks:
        if task == RejectOption: continue
        ret.update(get_single_task_metrics(task))
    # pretty hacky way of integrating reject option
    # todo: put back. Took out because of OOMing
    # if RejectOption in tasks:
    #     ret["select"] = RejectOptionMetric(cfg, deepcopy(ret))
    return ret

def reset_metrics(module):
    if isinstance(module, Metric):
        module.reset()
