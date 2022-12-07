import torch
import torch.nn.functional as F
from omegaconf.errors import ConfigAttributeError

from common.utils import get_activity

def act_ce_loss(batch, y_pred):
    return F.binary_cross_entropy_with_logits(y_pred, batch.is_active.float())

def act_mse_loss(batch, y_pred):
    return F.mse_loss(get_activity(batch), y_pred)

def energy_mse_loss(batch, y_pred):
    return F.mse_loss(batch.energy, y_pred.energy)

def coord_mse_loss(batch, y_pred):
    ret = []
    for lig, cp in zip(batch.lig, y_pred.lig_coord):
        ct = lig.ndata.coord
        ret.append(F.mse_loss(ct, cp))
    return torch.stack(ret).mean()

def pose_mse_loss(batch, y_pred):
    rot, trans = y_pred
    rot_true = torch.eye(3, device=rot.device).unsqueeze(0).repeat((rot.size(0),1,1))
    trans_true = torch.zeros(trans.size(0),3, device=trans.device)
    return F.mse_loss(rot_true, rot) + F.mse_loss(trans_true, trans)

def combine_losses(loss_cfg, loss_fns, *args):
    ret = 0
    terms = {}
    for fn in loss_fns:
        name = '_'.join(fn.__name__.split('_')[:-1])
        try:
            lam = getattr(loss_cfg, name + "_lambda")
        except ConfigAttributeError:
            lam = 0.0
        if lam:
            loss = fn(*args)
            ret += lam*loss
            terms[name] = loss
    return ret, terms

def get_losses(cfg, batch, y_pred):
    """ Returns both the total loss, and a dict mapping names of the loss functions
    to the loss value """
    losses = [ act_ce_loss, act_mse_loss, coord_mse_loss, pose_mse_loss ]
    loss_cfg = cfg.losses
    return combine_losses(loss_cfg, losses, batch, y_pred)