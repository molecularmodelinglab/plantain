# This file defines the types of data that could be returned by a dataset

import torch
from omegaconf import DictConfig

from terrace.batch import Batchable
from terrace.type_data import ClassTD, TensorTD
from datasets.graphs.mol_graph import MolGraph
from datasets.graphs.prot_graph import ProtGraph

class ActivityData(Batchable):
    lig: MolGraph
    rec: ProtGraph
    activity: torch.Tensor

    @staticmethod
    def get_type_data(cfg: DictConfig):
        lig_td = MolGraph.get_type_data(cfg)
        rec_td = ProtGraph.get_type_data(cfg)
        act_td = TensorTD((1,))
        return ClassTD(ActivityData, lig=lig_td, rec=rec_td, activity=act_td)