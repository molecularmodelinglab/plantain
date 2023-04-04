import torch
import torch.nn.functional as F
import dgl.backend as dF

def bce_loss(x, pred, y):
    return F.binary_cross_entropy_with_logits(pred, y.float())

def act_mse_loss(x, pred, y):
    mask = ~torch.isnan(y.activity)
    pred = pred.activity[mask]
    y = y.activity[mask]
    if mask.sum() == 0: return 0.0
    return F.mse_loss(pred, y)

def x_mse_loss(x, pred, y):
    return F.mse_loss(pred, x)

def bce_mse(x, pred, y):
    """ MSE between actual BCE and predicted BCE"""
    true_bce = F.binary_cross_entropy_with_logits(pred.active_prob_unnorm, y.is_active.float(), reduction='none').detach()
    return F.mse_loss(pred.pred_bce, true_bce)

def act_ce_loss(x, pred, y):
    y_ce = torch.cat((y.is_active.unsqueeze(-1), ~y.is_active.unsqueeze(-1)), -1)
    return F.cross_entropy(pred.softmax_logits, y_ce.float())

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

def gnina_docked_mse_loss(x, pred, y):
    true_score = torch.stack([scores[0] for scores in x.affinities])
    return F.mse_loss(pred.docked_score, true_score)

def worse_pose_bce_loss(x, pred, y):
    pred = pred.pose_scores[:,-1]
    y = torch.zeros_like(y.is_active)
    return F.binary_cross_entropy_with_logits(pred, y.float())

def crystal_pose_ce_loss(x, pred, y):
    p = pred.full_pose_scores
    assert p.shape[1] == 2
    labels = torch.ones((p.shape[0],), dtype=torch.long, device=p.device)
    return F.cross_entropy(p, labels)

def best_docked_ce_loss(x, pred, y):
    cutoff = torch.asarray([[2.0]]*len(y.pose_rmsds), device=y.pose_rmsds.device)
    all_rmsds = torch.cat((cutoff, y.pose_rmsds), -1)
    labels = torch.argmin(all_rmsds, -1)
    return F.cross_entropy(pred.full_pose_scores, labels)

def worst_pose_atn_ce_loss(x, pred, y):
    p = pred.full_pose_scores
    p = torch.cat((p[:,:-1].mean(-1, keepdims=True), p[:,-1:]), -1)
    labels = torch.zeros((p.shape[0],), dtype=torch.long, device=p.device)
    return F.cross_entropy(p, labels)

def opposite_pose_bce_loss(x, pred, y):
    pred = pred.pose_scores[:,-1]
    y = ~y.is_active
    return F.binary_cross_entropy_with_logits(pred, y.float())

def pose_class_bce_loss(x, pred, y):
    p0 = pred.pose_scores[:,0]
    p1 = pred.pose_scores[:,1]
    y0 = torch.ones_like(y.is_active)
    y1 = torch.zeros_like(y.is_active)
    l1 = F.binary_cross_entropy_with_logits(p0, y0.float())
    l2 = F.binary_cross_entropy_with_logits(p1, y1.float())
    return 0.5*l1 + 0.5*l2

def rot_mse(x, pred, y):
    yt = torch.zeros_like(pred.diffused_transforms.rot)
    return F.mse_loss(pred.diffused_transforms.rot, yt)

def trans_mse(x, pred, y):
    yt = torch.zeros_like(pred.diffused_transforms.trans)
    return F.mse_loss(pred.diffused_transforms.trans, yt)

def diffused_pose_mse(x, pred, y):
    ret = []
    for ppose, tpose in zip(pred.diffused_poses, y.lig_crystal_pose):
        yt = tpose.coord.repeat(ppose.coord.size(0),1,1)
        ret.append(F.mse_loss(ppose.coord, yt))
    return torch.stack(ret).mean()

def diffused_rmsd_mse(x, pred, y):
    return F.mse_loss(pred.diffused_energy, pred.diffused_rmsds)

def full_inv_dist_mse(x, pred, y):
    lig_coord = torch.cat(y.lig_crystal_pose.coord, 0)
    lig_coord = dF.pad_packed_tensor(lig_coord, x.lig_graph.dgl().batch_num_nodes(), 0.0)
    rec_coord = dF.pad_packed_tensor(x.full_rec_data.ndata.coord, x.full_rec_data.dgl().batch_num_nodes(), 0.0)
    dist = torch.cdist(lig_coord, rec_coord)
    mask = dist != 0.0
    true_mat = 1.0/(1.0 + dist)
    return F.mse_loss(pred.inv_dist_mat, true_mat, reduction='none')[mask].mean()

def get_single_loss(loss_cfg, x, pred, y):
    loss_fn = {
        "bce": bce_loss,
        "inv_dist_mse": inv_dist_mse,
        "rec_interaction_bce": rec_interaction_bce,
        "inv_dist_mean_std": inv_dist_mean_std,
        "bce_mse": bce_mse,
        "x_mse": x_mse_loss,
        "docked_mse": docked_mse_loss,
        "gnina_docked_mse": gnina_docked_mse_loss,
        "act_ce": act_ce_loss,
        "worse_pose_bce": worse_pose_bce_loss,
        "opposite_pose_bce": opposite_pose_bce_loss,
        "pose_class_bce": pose_class_bce_loss,
        "rot_mse": rot_mse,
        "trans_mse": trans_mse,
        "diffused_pose_mse": diffused_pose_mse,
        "diffused_rmsd_mse": diffused_rmsd_mse,
        "crystal_pose_ce": crystal_pose_ce_loss,
        "worse_pose_atn_ce": worst_pose_atn_ce_loss,
        "best_docked_ce": best_docked_ce_loss,
        "act_mse": act_mse_loss,
        "full_inv_dist_mse": full_inv_dist_mse,
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

def get_losses(cfg, task_names, x, pred, y):

    if "selective_net" in cfg.losses:
        return selective_net_loss(cfg.losses.selective_net, x, pred, y)
    elif "selective_softmax" in cfg.losses:
        return selective_softmax_loss(cfg.losses.selective_softmax, x, pred, y)
    elif "selective_softmax_pos" in cfg.losses:
        return selective_softmax_pos_loss(cfg.losses.selective_softmax_pos, x, pred, y)

    total_loss = 0.0
    ret = {}
    for loss_name, loss_cfg in cfg.losses.items():
        if "task" in loss_cfg:
            if loss_cfg.task is None:
                if len(task_names) != 0:
                    continue
            elif loss_cfg.task not in task_names:
                continue
        loss = get_single_loss(loss_cfg, x, pred, y)
        if loss_cfg.weight > 0.0:
            total_loss += loss*loss_cfg.weight
        ret[loss_name] = loss
    
    return total_loss, ret