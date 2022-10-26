import os
import random
import torch
from abc import ABC, abstractmethod
from meeko import PDBQTMolecule

from datasets.bigbind_screen import BigBindScreenDataset
from datasets.lit_pcba import LitPcbaDataset
from common.old_routine import get_old_model, old_model_key, get_weight_artifact

class ValModel(ABC):

    @abstractmethod
    def get_cache_key(self):
        return

    @abstractmethod
    def get_name(self):
        return

    @abstractmethod
    def __call__(self, x, dataset):
        return

    def to(self, device):
        return self

class OldModel(ValModel):

    def __init__(self, cfg, run, tag):
        self.run = run
        self.model = get_old_model(cfg, run, tag)
        self.key = old_model_key(cfg, run, tag)

    def get_cache_key(self):
        return self.key

    def get_name(self):
        artifact = get_weight_artifact(self.run)
        return f"{self.run.id}_{artifact.version}"

    def __call__(self, x, dataset):
        return self.model(x)

    def to(self, device):
        self.model = self.model.to(device)
        return self

class VinaModel(ValModel):

    def __init__(self, cfg):
        self.dir = cfg.platform.bigbind_docked_dir

    def get_cache_key(self):
        return "vina"

    def get_name(self):
        return "vina"

    def __call__(self, batch, dataset):
        assert isinstance(dataset, BigBindScreenDataset)
        ret = []
        for index in batch.index:
            pdbqt_file = f"{self.dir}/{dataset.split}_screens/{dataset.target}/{index}.pdbqt"
            if os.path.exists(pdbqt_file):
                ret.append(-PDBQTMolecule.from_file(pdbqt_file).score)
            else:
                ret.append(-100)
        return torch.tensor(ret, dtype=torch.float32, device=batch.index.device)

class GninaModel(ValModel):

    def __init__(self, cfg):
        self.pcba_scores = {}
        with open("./prior_work/newdefault_CNNaffinity-max.summary") as f:
            for line in f.readlines():
                _, score, target, idx, _ = line.split()
                self.pcba_scores[(target, int(idx))] = float(score)

    def get_cache_key(self):
        return "gnina"

    def get_name(self):
        return "gnina"

    def __call__(self, batch, dataset):
        assert isinstance(dataset, LitPcbaDataset)
        ret = []
        for index in batch.index:
            key = (dataset.target, int(index))
            if key in self.pcba_scores:
                ret.append(self.pcba_scores[key])
            else:
                ret.append(-100)
        return torch.tensor(ret, dtype=torch.float32, device=batch.index.device)

class ComboModel(ValModel):

    def __init__(self, model1: ValModel, model2: ValModel, model1_frac: float):
        self.model1 = model1
        self.model2 = model2
        self.model1_frac = model1_frac
        self.model1_preds = None
        self.model2_preds = None

    def get_cache_key(self):
        return ("combo", self.model1.get_cache_key(), self.model2.get_cache_key(), self.model1_frac)
    
    def get_name(self):
        return f"combo_{self.model1.get_name()}_{self.model2.get_name()}_{self.model1_frac}"

    def init_preds(self, model1_preds, model2_preds):
        self.model1_preds = model1_preds
        self.model2_preds = model2_preds

    def __call__(self, x, dataset):
        raise NotImplementedError()

    def choose_topk(self, k):
        """ returns the indexes of the top k items according to our choice
        hueristic (top model1_frac from model1, top k from those) """
        idx_p1_p2 = list(zip(range(len(self.model1_preds)), self.model1_preds, self.model2_preds))
        random.shuffle(idx_p1_p2)
        k1 = int(len(self.model1_preds)*self.model1_frac)
        top_k1 = sorted(idx_p1_p2, key=lambda x: -x[1])[:k1]
        top_k = sorted(top_k1, key=lambda x: -x[2])[:k]
        return [ idx for idx, p1, p2 in top_k ]