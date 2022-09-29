from torch import nn
from torchmetrics import Metric, Accuracy, AUROC

class MetricWrapper(Metric):

    def __init__(self, metric, get_y):
        super().__init__()
        self.metric = metric
        self.get_y = get_y

    def update(self, pred, batch):
        y_true = self.get_y(batch)
        return self.metric.update(pred, y_true)

    def compute(self):
        return self.metric.compute()

def get_metrics(cfg):
    return {
        "classification": nn.ModuleDict({
            "acc": MetricWrapper(Accuracy(), lambda b: b.is_active),
            "bal_acc": MetricWrapper(Accuracy(average="macro", num_classes=2, multiclass=True), lambda b: b.is_active),
            "auroc": MetricWrapper(AUROC(), lambda b: b.is_active)
        })
    }[cfg.task]
