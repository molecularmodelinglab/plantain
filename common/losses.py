
import torch.nn.functional as F

from common.utils import get_activity

def act_mse_loss(batch, y_pred):
    return F.mse_loss(get_activity(batch), y_pred)

def combine_losses(loss_cfg, loss_fns, *args):
    ret = 0
    terms = {}
    for fn in loss_fns:
        loss = fn(*args)
        name = '_'.join(fn.__name__.split('_')[:-1])
        lam = getattr(loss_cfg, name + "_lambda")
        if lam:
            ret += lam*loss
            terms[name] = loss
    return ret, terms

def get_losses(cfg, batch, y_pred):
    """ Returns both the total loss, and a dict mapping names of the loss functions
    to the loss value """
    losses = [ act_mse_loss ]
    loss_cfg = cfg.losses
    return combine_losses(loss_cfg, losses, batch, y_pred)