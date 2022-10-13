import os
import torch
from abc import ABC, abstractmethod
from meeko import PDBQTMolecule

from datasets.bigbind_screen import BigBindScreenDataset
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