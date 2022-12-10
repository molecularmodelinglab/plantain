import torch
import torch.nn.functional as F

def get_rmsds(batch, pred_coords):
    # not symmetric corrected - change!
    rmsds = []
    for b, pred in zip(batch, pred_coords):  
        true_coord = b.lig.ndata.coord 
        rmsds.append(torch.sqrt(F.mse_loss(true_coord, pred)))
    return torch.stack(rmsds)

def get_pose_acc(rmsds, cutoff):
    return sum(rmsds < cutoff)/len(rmsds)

def get_pose_metrics(prefix, batch_idx, model, batch):
    if not model.cfg.test_inference: return {
        "rmsd": 0,
        "acc_2": 0,
        "acc_5": 0
    }

    # inference is expensive so don't do it all the time
    if prefix == "train" and batch_idx % 10 != 0:
        return {}

    ret = {}
    with torch.no_grad():
        pred_coords = model.infer(batch)
        rmsds = get_rmsds(batch, pred_coords)
        ret["rmsd"] = rmsds.mean()
        ret["acc_2"] = get_pose_acc(rmsds, 2)
        ret["acc_5"] = get_pose_acc(rmsds, 5)
    return ret

def get_task_metrics(task, prefix, batch_idx, model, batch):
    if task == "pose":
        return get_pose_metrics(prefix, batch_idx, model, batch)
    return {}
