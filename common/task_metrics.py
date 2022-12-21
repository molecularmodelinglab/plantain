from collections import defaultdict
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from terrace.batch import make_batch

from datasets.bigbind_vina_struct import BigBindVinaStructDataset

def get_rmsds(batch, pred_coords):
    # not symmetry corrected - change!
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
    pred_coords = model.infer(batch)
    rmsds = get_rmsds(batch, pred_coords)
    ret["rmsd"] = rmsds.mean()
    ret["acc_2"] = get_pose_acc(rmsds, 2)
    ret["acc_5"] = get_pose_acc(rmsds, 5)
    return ret

def randomized_argsort(scores):
    """ Like argsort, but randomizes so that there's no change duplicate values 
    will make our metrics more optimistic.

    For instance, torch.argsort(torch.zeros(...)) will generally return ordered
    indexes (this isn't garunteed, but is generally the case). This function will
    return random indexes """

    idx = torch.randperm(scores.shape[0], device=scores.device)
    scores_perm = scores[idx].view(scores.size())
    return torch.argsort(scores_perm)[idx]

def get_pose_rank_metrics(cfg, model, split, num_iter):
    
    device = next(iter(model.parameters())).device
    dataset = BigBindVinaStructDataset(cfg, split)
    
    metrics = defaultdict(list)
    for i, d in enumerate(tqdm(dataset)):

        batch = make_batch([d]).to(device)
        # this is how we would score for vina
        # scores = -torch.arange(len(d.rmsds))

        scores = model.get_conformer_scores(batch)[0,:,0]
        # scores = torch.randn_like(d.rmsds)
        
        correct = d.rmsds < 2.0

        idx = randomized_argsort(-scores)
        correct_sorted = correct[idx]
        scores_sorted = scores[idx]

        if True in correct and False in correct:
            metrics["auroc"].append(roc_auc_score(correct_sorted.cpu(), scores_sorted.cpu()))

        for n in range(1,10):
            metrics[f"top_{n}_acc"].append(int(True in correct_sorted[:n]))
        if num_iter is not None and i > num_iter:
            break

    return { f"pose_{key}": torch.tensor(val, dtype=torch.float).mean().item() for key, val in metrics.items()}

def get_task_metrics(task, prefix, batch_idx, model, batch):
    """ Get task metrics for a single batch """
    with torch.no_grad():
        if task == "pose":
            return get_pose_metrics(prefix, batch_idx, model, batch)
        return {}

def get_epoch_task_metrics(cfg, task, model, split, num_iter):
    """ Get the task metrics for a whole epoch. Evaluates the model
    on whatever dataset you want (for num_iter steps. If num_iter is
    None, evals on the whole dataset). The dataset split is defined
    by the split arg """
    with torch.no_grad():
        if task == "pose_rank":
            return get_pose_rank_metrics(cfg, model, split, num_iter)
        return {}