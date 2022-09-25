import torch
import torch.nn.functional as F

from common.utils import get_activity

def act_mse_loss(batch, y_pred):
    return F.mse_loss(get_activity(batch), y_pred)

def coord_mse_loss(batch, y_pred):
    ret = []
    for lig, cp in zip(batch.lig, y_pred.lig_coord):
        ct = lig.ndata.coord
        ret.append(F.mse_loss(ct, cp))
    return torch.stack(ret).mean()

def combine_losses(loss_cfg, loss_fns, *args):
    ret = 0
    terms = {}
    for fn in loss_fns:
        name = '_'.join(fn.__name__.split('_')[:-1])
        lam = getattr(loss_cfg, name + "_lambda")
        if lam:
            loss = fn(*args)
            ret += lam*loss
            terms[name] = loss
    return ret, terms

def get_losses(cfg, batch, y_pred):
    """ Returns both the total loss, and a dict mapping names of the loss functions
    to the loss value """
    losses = [ act_mse_loss, coord_mse_loss ]
    loss_cfg = cfg.losses
    return combine_losses(loss_cfg, losses, batch, y_pred)