# This file defines the types of data that could be returned by a dataset

import torch
from omegaconf import DictConfig

from terrace.batch import Batchable
from terrace.type_data import ClassTD, TensorTD, ShapeVar
from datasets.graphs.mol_graph import MolGraph
from datasets.graphs.prot_graph import ProtGraph

class StructData(Batchable):
    lig: MolGraph
    rec: ProtGraph

    @staticmethod
    def get_type_data(cfg: DictConfig):
        lig_td = MolGraph.get_type_data(cfg)
        rec_td = ProtGraph.get_type_data(cfg)
        act_td = TensorTD((1,))
        return ClassTD(StructData, lig=lig_td, rec=rec_td)

class ActivityData(StructData):
    lig: MolGraph
    rec: ProtGraph
    activity: torch.Tensor

    @staticmethod
    def get_type_data(cfg: DictConfig):
        lig_td = MolGraph.get_type_data(cfg)
        rec_td = ProtGraph.get_type_data(cfg)
        act_td = TensorTD((1,))
        return ClassTD(ActivityData, lig=lig_td, rec=rec_td, activity=act_td)

class EnergyData(StructData):
    lig: MolGraph
    rec: ProtGraph
    energy: torch.Tensor

    @staticmethod
    def get_type_data(cfg: DictConfig):
        lig_td = MolGraph.get_type_data(cfg)
        rec_td = ProtGraph.get_type_data(cfg)
        en_td = TensorTD((1,))
        return ClassTD(ActivityData, lig=lig_td, rec=rec_td, energy=en_td)

class PredData(Batchable):

    # This is a weird case for a batchable class, since coord will have different
    # shapes depending on the number of nodes. Thus the Batch will contain lists,
    # not tensors directly. In the future, Batch should support this directly. But
    # for now it doesn't matter since we only ever go from Batch[PredData] -> PredData,
    # not the other way around
    lig_coord: torch.Tensor
    activity: torch.Tensor

class EnergyPredData(Batchable):
    lig_coord: torch.Tensor
    energy: torch.Tensor
