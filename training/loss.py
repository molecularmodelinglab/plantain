import torch
import torch.nn.functional as F

def bce_loss(x, pred, y):
    return F.binary_cross_entropy_with_logits(pred, y.float())

def x_mse_loss(x, pred, y):
    return F.mse_loss(pred, x)

def bce_mse(x, pred, y):
    """ MSE between actual BCE and predicted BCE"""
    true_bce = F.binary_cross_entropy_with_logits(pred.active_prob_unnorm, y.is_active.float(), reduction='none').detach()
    return F.mse_loss(pred.pred_bce, true_bce)

def inv_dist_mse(x, pred, y):
    losses = []
    for rec, lig_coords, pred_mat in zip(x.rec_graph, y.lig_coords, pred.inv_dist_mat):
        true_mat = 1.0/torch.cdist(lig_coords, rec.ndata.coord)
        loss = F.mse_loss(pred_mat, true_mat)
        losses.append(loss)
    return torch.stack(losses).mean()

def inv_dist_mean_std(x, pred, y):
    losses = []
    for inv_dist_mat in pred.inv_dist_mat:
        loss = 1.0/(1.0 + (inv_dist_mat/inv_dist_mat.sum(0)).std(0)).mean()
        loss += 1.0/(1.0 + (inv_dist_mat/inv_dist_mat.sum(1).unsqueeze(1)).std(1)).mean()
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

def docked_mse_loss(x, pred, y):
    true_score = torch.stack([scores[0] for scores in x.docked_scores])
    return F.mse_loss(pred.docked_score, true_score)

def get_single_loss(loss_cfg, x, pred, y):
    loss_fn = {
        "bce": bce_loss,
        "inv_dist_mse": inv_dist_mse,
        "rec_interaction_bce": rec_interaction_bce,
        "inv_dist_mean_std": inv_dist_mean_std,
        "bce_mse": bce_mse,
        "x_mse": x_mse_loss,
        "docked_mse": docked_mse_loss,
    }[loss_cfg.func]
    if "x" in loss_cfg:
        x = getattr(x, loss_cfg["x"])
    if "pred" in loss_cfg:
        pred = getattr(pred, loss_cfg["pred"])
    if "y" in loss_cfg:
        y = getattr(y, loss_cfg["y"])
    return loss_fn(x, pred, y)

def selective_net_loss(loss_cfg, x, pred, y):
    bce = F.binary_cross_entropy_with_logits(pred.active_prob_unnorm, y.is_active.float(), reduction='none')
    selective_bce = bce*pred.select_prob
    coverage = pred.select_prob.mean()
    cov_diff = loss_cfg.coverage - coverage
    cov_loss = 0.0 if cov_diff < 0 else cov_diff**2

    tot = loss_cfg.bce_weight*bce.mean() + (loss_cfg.selective_weight*selective_bce).mean() + loss_cfg.coverage_weight*cov_loss

    return tot, {
        "bce": bce.mean(),
        "selective_bce": selective_bce.mean(),
        "coverage": coverage,
        "coverage_loss": cov_loss
    }

def selective_softmax_loss(loss_cfg, x, pred, y):
    bce = F.binary_cross_entropy_with_logits(pred.active_prob_unnorm, y.is_active.float(), reduction='none')
    select_norm = torch.softmax(pred.select_unnorm, 0)
    selective_bce = (select_norm*bce).sum()
    tot = (1.0-loss_cfg.alpha)*bce.mean() + loss_cfg.alpha*selective_bce
    return tot, {
        "bce": bce.mean(),
        "selective_bce": selective_bce,
    }

def selective_softmax_pos_loss(loss_cfg, x, pred, y):
    bce = F.binary_cross_entropy_with_logits(pred.active_prob_unnorm, y.is_active.float(), reduction='none')
    if y.is_active.sum() > 0:
        select_norm = torch.softmax(pred.select_unnorm[y.is_active], 0)
        selective_bce = (select_norm*bce[y.is_active]).sum()
    else:
        selective_bce = 0.0
    tot = (1.0-loss_cfg.alpha)*bce.mean() + loss_cfg.alpha*selective_bce
    return tot, {
        "bce": bce.mean(),
        "selective_bce": selective_bce,
    }

def get_losses(cfg, tasks, x, pred, y):

    if "selective_net" in cfg.losses:
        return selective_net_loss(cfg.losses.selective_net, x, pred, y)
    elif "selective_softmax" in cfg.losses:
        return selective_softmax_loss(cfg.losses.selective_softmax, x, pred, y)
    elif "selective_softmax_pos" in cfg.losses:
        return selective_softmax_pos_loss(cfg.losses.selective_softmax_pos, x, pred, y)

    task_names = [ task.get_name() for task in tasks ]
    total_loss = 0.0
    ret = {}
    for loss_name, loss_cfg in cfg.losses.items():
        if "task" in loss_cfg:
            if loss_cfg.task not in task_names:
                continue
        loss = get_single_loss(loss_cfg, x, pred, y)
        if loss_cfg.weight > 0.0:
            total_loss += loss*loss_cfg.weight
        ret[loss_name] = loss
    
    return total_loss, ret