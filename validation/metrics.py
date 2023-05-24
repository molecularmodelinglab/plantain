from collections import defaultdict
from copy import deepcopy
import warnings
from rdkit.Chem.rdMolAlign import CalcRMS, GetBestRMS
import random

from tqdm import tqdm
from terrace import Batch, collate
from typing import Callable, Dict, List, Set, Type
import torch
from torch import nn
from torchmetrics import Metric, SpearmanCorrCoef, Accuracy, AUROC, ROC, Precision, R2Score, MeanSquaredError
from terrace.dataframe import DFRow
from common.pose_transform import add_pose_to_mol

class FullMetric(Metric):
    """ Extension of torchmetrics metrics which also allows us
    to update based on input batches """

    def update(self, x: Batch[DFRow], pred: Batch[DFRow], label: Batch[DFRow]):
        raise NotImplementedError()

    def compute(self):
        raise NotImplementedError

class MetricWrapper(FullMetric):

    def __init__(self, metric_class: Type[Metric], get_y_p: Callable, get_y_t: Callable, *args, **kwargs):
        super().__init__()
        self.metric = metric_class(*args, **kwargs)
        self.get_y_p = get_y_p
        self.get_y_t = get_y_t

    def update(self, x: Batch[DFRow], pred: Batch[DFRow], label: Batch[DFRow]):
        y_pred = self.get_y_p(pred)
        y_true = self.get_y_t(label)
        return self.metric.update(y_pred, y_true)

    def compute(self):
        return self.metric.compute()

def dict_stack(items):
    if isinstance(items, list):
        ex = items[0]
        if isinstance(ex, dict):
            return {key: dict_stack([item[key] for item in items]) for key in ex.keys()}
    return torch.stack(items)

def dict_apply(f, d):
    if isinstance(d, dict):
        return {key: f(val) for key, val in d.items()}
    return f(d)

class PerPocketMetric(FullMetric):
    """ Computes a specific metric for every pocket and returns the 
    mean and median values """

    def __init__(self, metric_class: Type[FullMetric], *args, **kwargs):
        """ reduce is either mean or median"""
        super().__init__()
        self.metric_maker = lambda: metric_class(*args, **kwargs)
        self.metric = self.metric_maker()
        self.pocket_metrics = nn.ModuleDict()

    def reset(self):
        self.metric.reset()
        self.pocket_metrics = nn.ModuleDict()

    def update(self, x_b: Batch[DFRow], pred_b: Batch[DFRow], label_b: Batch[DFRow]):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            self.metric.update(x_b, pred_b, label_b)
            for x, pred, label in zip(x_b, pred_b, label_b):
                poc_id = x.pocket_id
                xi = collate([x], lazy=True)
                predi = collate([pred], lazy=True)
                labeli = collate([label], lazy=True)
                if poc_id not in self.pocket_metrics:
                    self.pocket_metrics[poc_id] = self.metric_maker().to(self.device)
                self.pocket_metrics[poc_id].update(xi, predi, labeli)

    def pre_compute(self):
        pass

    def compute(self):
        self.pre_compute()
        results = []
        for metric in self.pocket_metrics.values():
            results.append(metric.compute())
        if len(results) == 0:
            return {
                "all": self.metric.compute()
            }
        else:
            results = dict_stack(results)
            return {
                "all": self.metric.compute(),
                "mean": dict_apply(lambda r: r.mean(), results),
                "median": dict_apply(lambda r: r.median(), results)
            }

class PosePerPocketMetric(PerPocketMetric):

    def update(self, x, pred, y):
        if not hasattr(pred, "lig_pose"): return
        super().update(x, pred, y)

# needed because the pickle can't handle lambdas
def get_is_active(b):
    return b.is_active

def get_is_active_score(p):
    return p.is_active_score

def get_activity_score(p):
    return p.activity_score

def get_active_prob(p):
    return p.active_prob

def get_activity(b):
    return b.activity

def identity(b):
    return b

class IsActiveAUC(MetricWrapper):

    def __init__(self):
        super().__init__(AUROC, get_is_active_score, get_is_active, 'binary')

    def reset(self):
        super().reset()

class PerPocketAUC(PerPocketMetric):

    def __init__(self):
        super().__init__(IsActiveAUC)

    def pre_compute(self):
        """ Can't compute AUC for labels that are all active or all inactive"""
        for key, val in list(self.pocket_metrics.items()):
            if True not in val.metric.target or False not in val.metric.target:
                del self.pocket_metrics[key]

class ActivitySpearman(MetricWrapper):

    def __init__(self):
        super().__init__(SpearmanCorrCoef, get_activity_score, get_activity)

class PoseRankAcc(FullMetric):

    def __init__(self, rmsd_cutoff=2.0):
        super().__init__()
        self.rmsd_cutoff = rmsd_cutoff
        self.top_n_correct = defaultdict(int)
        self.total_seen = 0

    def reset(self) -> None:
        self.top_n_correct = defaultdict(int)
        self.total_seen = 0
        
    def update(self, x, pred, y):
        for pred0, y0 in zip(pred, y):
            # if torch.amax(pred0.pose_scores) < -0.25:
            #     print("skipping")
            #     continue
            # print("doing!")
            poses_correct = y0.pose_rmsds < self.rmsd_cutoff
            correct_sorted = [ x[0].cpu().item() for x in sorted(zip(poses_correct, -pred0.pose_scores, torch.randn_like(pred0.pose_scores)), key=lambda x: x[1:])]
            for n in range(1, len(correct_sorted)+1):
                self.top_n_correct[n] += int(True in correct_sorted[:n])
            self.total_seen += 1

    def compute(self):
        # print("TOTAL: ", self.total_seen)
        ret = {}
        for n, correct in self.top_n_correct.items():
            ret[f"top_{n}"] = correct/self.total_seen
        return ret

def get_rmsds(ligs, pred_pose, true_pose, align=False):
    rms_func = GetBestRMS if align else CalcRMS
    ret = []
    for lig, pred, yt in zip(ligs, pred_pose, true_pose):
        mol1 = deepcopy(lig)
        mol2 = deepcopy(lig)
        add_pose_to_mol(mol2, yt)
        if pred.coord.dim() == 3:
            ret.append([])
            for p in pred.items():
                add_pose_to_mol(mol1, p)
                ret[-1].append(rms_func(mol1, mol2, maxMatches=1000))
        else:
            add_pose_to_mol(mol1, pred)
            # made maxMatches very low for now because it's freezing training
            # with the default value. Perhaps raise later
            ret.append(rms_func(mol1, mol2, maxMatches=1000))
    return torch.asarray(ret, dtype=torch.float32, device=pred_pose[0].coord.device)

class PoseRMSD(FullMetric):

    def __init__(self):
        super().__init__()
        self.add_state("rmsd_sum", default=torch.tensor(0.0))
        self.add_state("total", default=torch.tensor(0))

    def update(self, x, pred, y):
        # if not hasattr(pred, "lig_pose"): return
        rmsds = get_rmsds(x.lig, pred.lig_pose, y.lig_crystal_pose)
        if rmsds.dim() == 2:
            rmsds = rmsds[:,0]
        self.rmsd_sum += rmsds.sum()
        self.total += len(x)

    def compute(self):
        # if self.total == 0:
        #     print("Something is wrong with PoseRMSD metric. I haven't been updated.")
        assert self.total > 0
        return self.rmsd_sum / self.total

class PoseAcc(FullMetric):

    def __init__(self, rmsd_cutoff):
        super().__init__()
        self.rmsd_cutoff = rmsd_cutoff
        self.correct_n = defaultdict(int)
        self.total_n = defaultdict(int)

    def reset(self):
        self.correct_n = defaultdict(int)
        self.total_n = defaultdict(int)

    def update(self, x, pred, y):
        # if not hasattr(pred, "lig_pose"): return
        rmsds = get_rmsds(x.lig, pred.lig_pose, y.lig_crystal_pose)
        if rmsds.dim() == 1:
            rmsds = rmsds.unsqueeze(1)
        correct = (rmsds < self.rmsd_cutoff)
        for n in range(correct.shape[1]):
            self.correct_n[n] += correct[:,:n+1].amax(1).sum()
            self.total_n[n] += len(x)

    def compute(self):
        ret = {}
        for n in self.total_n:
            ret[f"{n+1}"] = self.correct_n[n].float()/self.total_n[n]
        return ret

class CrystalEnergy(FullMetric):
    
    def __init__(self, rmsd_cutoff):
        super().__init__()
        self.rmsd_cutoff = rmsd_cutoff
        self.add_state("num_local", default=torch.tensor(0))
        self.add_state("num_global", default=torch.tensor(0))
        self.add_state("total", default=torch.tensor(0))

    def update(self, x, pred, y):
        if not hasattr(pred, "crystal_pose"): return
        crystal_rmsds = get_rmsds(x.lig, pred.crystal_pose, y.lig_crystal_pose)
        local = crystal_rmsds < self.rmsd_cutoff
        self.num_local += local.sum()
        for loc, energy, crys_energy in zip(local, pred.energy, pred.crystal_energy):
            if not loc: continue
            all_energies = torch.cat((energy, crys_energy.unsqueeze(0)), 0)
            if all_energies.min() == crys_energy:
                self.num_global += 1
        self.total += len(x)

    def compute(self):
        if self.total == 0: return {}
        return {
            "global_min": self.num_global.float()/self.total,
            "local_min": self.num_local.float()/self.total
        }

class EnrichmentFactor(FullMetric):

    def __init__(self):
        super().__init__()
        self.scores = []
        self.yts = []

    def update(self, x, pred, y):
        for pred0, y0 in zip(pred, y):
            self.yts.append(y0.is_active)
            self.scores.append(pred0.is_active_score)

    def reset(self):
        super().reset()
        self.yts = []
        self.scores = []

    def compute(self):
        yt, scores = collate(self.yts), collate(self.scores)

        one_percent = round(len(yt)*0.01)
        if one_percent == 0: return {}

        all_pred_and_act = list(zip(scores, yt))
        random.shuffle(all_pred_and_act)
        pred_and_act = sorted(all_pred_and_act, key=lambda x: -x[0])[:one_percent]
        are_active = [ item[1] for item in pred_and_act ]
        tot_actives = sum(yt)
        max_actives = min(tot_actives, one_percent)
        frac_act_chosen = sum(are_active)/len(are_active)
        max_act_frac = max_actives/len(are_active)
        frac_act_in_set = tot_actives/len(yt)
        ef1 = frac_act_chosen/frac_act_in_set
        max_ef1 = max_act_frac/frac_act_in_set
        nef1 = ef1/max_ef1
        return {
            "ef1": ef1,
            "nef1": nef1,
            "total_in_set": len(yt),
            "total_chosen": one_percent,
            "total_actives_chosen": sum(are_active),
            "total_actives_in_set": tot_actives,
        }

class RejectOptionMetric(FullMetric):

    def __init__(self, cfg, metrics):
        super().__init__()
        self.select_fracs = torch.arange(0.05, 1.05, 0.05)
        self.metrics = nn.ModuleDict(metrics)
        self.xs = []
        self.ys = []
        self.preds = []
        self.dummy_param = nn.Parameter(torch.empty(0))

    def update(self, x, pred, y):
        for x0, pred0, y0 in zip(x, pred, y):
            self.xs.append(x0)
            self.ys.append(y0)
            self.preds.append(pred0)

    def reset(self):
        super().reset()
        self.xs = []
        self.ys = []
        self.preds = []

    def compute(self):
        device = self.dummy_param.device
        ret = {}
        x, y, pred = collate(self.xs).to(device), collate(self.ys).to(device), collate(self.preds).to(device)
        select_score = -pred.select_score
        idx = torch.argsort(select_score, dim=0)
        prev_idx = 0
        for frac in self.select_fracs:
            length = int(len(idx)*frac)
            if length == prev_idx: continue
            for index in idx[prev_idx:length]:
                x0 = collate([x[index]])
                y0 = collate([y[index]])
                pred0 = collate([pred[index]])
                for metric in self.metrics.values():
                    metric.update(x0, pred0, y0)
            prev_idx = length

            ret[f"{frac:.2f}"] = {
                key: val.compute() for key, val in self.metrics.items()
            }
        return ret

class ActivityR2(FullMetric):

    def __init__(self):
        super().__init__()
        self.metric = R2Score()

    def update(self, x, pred, y):
        mask = ~torch.isnan(y.activity)
        pred = pred.activity[mask]
        y = y.activity[mask]
        self.metric.update(pred, y)

    def compute(self):
        if self.metric.total < 2:
            return torch.tensor(torch.nan, device=self.metric.sum_error.device)
        return self.metric.compute()


class PerPocketActR2(PerPocketMetric):

    def __init__(self):
        super().__init__(ActivityR2)

    def pre_compute(self):
        """ Can't compute AUC for labels that are all active or all inactive"""
        for key, val in list(self.pocket_metrics.items()):
            if val.metric.total < 2:
                del self.pocket_metrics[key]

def get_single_task_metrics(task: str):
    return {
        "score_activity_class": nn.ModuleDict({
            "auroc": PerPocketAUC()
        }),
        "score_activity_regr": nn.ModuleDict({
            "spearman": PerPocketMetric(ActivitySpearman),
        }),
        "classify_activity": nn.ModuleDict({
            "acc": MetricWrapper(Accuracy, get_active_prob, get_is_active, 'binary'),
            "precision": MetricWrapper(Precision, get_active_prob, get_is_active, 'binary')
        }),
        "score_pose": nn.ModuleDict({
            "acc_2": PoseRankAcc(2.0),
            "acc_5": PoseRankAcc(5.0)
        }),
        "predict_lig_pose": nn.ModuleDict({
            "rmsd": PosePerPocketMetric(PoseRMSD),
            "acc_2": PosePerPocketMetric(PoseAcc, 2.0),
            "acc_5": PosePerPocketMetric(PoseAcc, 5.0),
            # "crystal_2": CrystalEnergy(2.0),
            # "crystal_5": CrystalEnergy(5.0)
        }),
        "predict_activity": nn.ModuleDict({
            "r2": PerPocketActR2(),
        })
    }[task]

def get_metrics(cfg, tasks: List[str], offline=False):
    # filter out those torchmetrics warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)

        ret = nn.ModuleDict({})
        for task in tasks:
            if task == "reject_option": continue
            ret.update(get_single_task_metrics(task))

        if offline and "score_activity_class" in tasks:
            ret.update({"enrichment": EnrichmentFactor()})
        
        # pretty hacky way of integrating reject option
        # todo: put back. Took out because of OOMing
        if "reject_option" in tasks and offline:
            ret["select"] = RejectOptionMetric(cfg, deepcopy(ret))
        
    return ret

def reset_metrics(module):
    if isinstance(module, Metric):
        module.reset()
