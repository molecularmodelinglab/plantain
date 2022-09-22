import torch.nn.functional as F

from common.utils import get_activity

def act_r2(batch, y_pred, variance_dict):
    return 1.0 - F.mse_loss(get_activity(batch), y_pred)/variance_dict["activity"]

def get_metrics(cfg, batch, y_pred, variance_dict):
    ret = {}
    for metric_name in cfg.metrics:
        # unsafe, but I'm taking taking some random cfg file from the internet
        f = globals()[metric_name]
        ret[metric_name] = f(batch, y_pred, variance_dict)
    return ret