import torch
from torch import nn
from torchmetrics import Metric, Accuracy, AUROC, ROC, Precision
# rom torchmetrics.classification import BinaryPrecision

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

def get_metrics(cfg):
    return {
        "classification": nn.ModuleDict({
            "acc": MetricWrapper(Accuracy(), torch.sigmoid, lambda b: b.is_active),
            "bal_acc": MetricWrapper(Accuracy(average="macro", num_classes=2, multiclass=True), torch.sigmoid, lambda b: b.is_active),
            "auroc": MetricWrapper(AUROC(), torch.sigmoid, lambda b: b.is_active),
            "precision": MetricWrapper(Precision(), torch.sigmoid, lambda b: b.is_active),
            "roc": MetricWrapper(ROC(), torch.sigmoid, lambda b: b.is_active)
        })
    }[cfg.task]
