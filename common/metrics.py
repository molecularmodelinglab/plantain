import torch
import torch.nn.functional as F

from common.utils import get_activity

def act_r2(batch, y_pred, variance_dict):
    return 1.0 - F.mse_loss(get_activity(batch), y_pred)/variance_dict["activity"]

def coord_rmsd(batch, y_pred, variance_dict):
    ret = []
    for lig, cp in zip(batch.lig, y_pred.lig_coord):
        ct = lig.ndata.coord
        ret.append(torch.sqrt(F.mse_loss(ct, cp)))
    return torch.stack(ret).mean()

def get_metrics(cfg, batch, y_pred, variance_dict):
    ret = {}
    for metric_name in cfg.metrics:
        # unsafe, but I'm not taking taking some random cfg file from the internet
        f = globals()[metric_name]
        ret[metric_name] = f(batch, y_pred, variance_dict)
    return ret