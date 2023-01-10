import torch
import torch.nn.functional as F

def get_single_loss(loss_cfg, pred, y):
    loss_fn = {
        "bce": F.binary_cross_entropy_with_logits
    }[loss_cfg.func]

    if "pred" in loss_cfg:
        pred = getattr(pred, loss_cfg["pred"])
    if "y" in loss_cfg:
        y = getattr(y, loss_cfg["y"])
    return loss_fn(pred, y.float())

def get_losses(cfg, pred, y):
    total_loss = 0.0
    ret = {}
    for loss_name, loss_cfg in cfg.losses.items():
        loss = get_single_loss(loss_cfg, pred, y)
        if loss_cfg.weight > 0.0:
            total_loss += loss
        ret[loss_name] = loss
    return total_loss, ret