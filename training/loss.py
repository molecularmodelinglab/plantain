import torch
import torch.nn.functional as F

def bce_loss(x, pred, y):
    return F.binary_cross_entropy_with_logits(pred, y.float())

def inv_dist_mse(x, pred, y):
    losses = []
    for rec, lig_coords, pred_mat in zip(x.rec_graph, y.lig_coords, pred.inv_dist_mat):
        true_mat = 1.0/torch.cdist(lig_coords, rec.ndata.coord)
        loss = F.mse_loss(pred_mat, true_mat)
        losses.append(loss)
    return torch.stack(losses).mean()

def rec_interaction_bce(x, pred, y):
    
    losses = []
    for lig_graph, rec_graph, pred_mat in zip(x.lig_graph, x.rec_graph, pred.inv_dist_mat):
        lig_coord = lig_graph.ndata.coord
        rec_coord = rec_graph.ndata.coord
        dist = torch.cdist(lig_coord, rec_coord)

        nulls = torch.tensor([[5.0]]*len(lig_coord), device=lig_coord.device)
        dist = torch.cat((dist, nulls), 1)
        labels = torch.argmin(dist, 1)

        losses.append(F.cross_entropy(pred_mat, labels))

    return torch.stack(losses).mean()


def get_single_loss(loss_cfg, x, pred, y):
    loss_fn = {
        "bce": bce_loss,
        "inv_dist_mse": inv_dist_mse,
        "rec_interaction_bce": rec_interaction_bce,
    }[loss_cfg.func]
    if "x" in loss_cfg:
        x = getattr(x, loss_cfg["x"])
    if "pred" in loss_cfg:
        pred = getattr(pred, loss_cfg["pred"])
    if "y" in loss_cfg:
        y = getattr(y, loss_cfg["y"])
    return loss_fn(x, pred, y)

def get_losses(cfg, x, pred, y):
    total_loss = 0.0
    ret = {}
    for loss_name, loss_cfg in cfg.losses.items():
        loss = get_single_loss(loss_cfg, x, pred, y)
        if loss_cfg.weight > 0.0:
            total_loss += loss
        ret[loss_name] = loss
    return total_loss, ret