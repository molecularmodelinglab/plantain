# This file defines the types of data that could be returned by a dataset

import torch
from omegaconf import DictConfig

from terrace.batch import Batchable
from terrace.type_data import ClassTD, TensorTD, ShapeVar
from datasets.graphs.mol_graph import MolGraph
from datasets.graphs.prot_graph import ProtGraph
from datasets.graphs.interaction_graph import InteractionGraph

class StructData(Batchable):
    lig: MolGraph
    rec: ProtGraph

    @staticmethod
    def get_type_data(cfg: DictConfig):
        lig_td = MolGraph.get_type_data(cfg)
        rec_td = ProtGraph.get_type_data(cfg)
        return ClassTD(StructData, lig=lig_td, rec=rec_td)

class ActivityData(StructData):
    lig: MolGraph
    rec: ProtGraph
    activity: torch.Tensor
    is_active: torch.Tensor

    @staticmethod
    def get_type_data(cfg: DictConfig):
        lig_td = MolGraph.get_type_data(cfg)
        rec_td = ProtGraph.get_type_data(cfg)
        act_td = TensorTD((1,))
        is_act_td = TensorTD((1,), dtype=bool)
        return ClassTD(ActivityData, lig=lig_td, rec=rec_td, activity=act_td, is_active=is_act_td)

class InteractionActivityData(StructData):

    graph: InteractionGraph
    is_active: torch.Tensor

    @staticmethod
    def get_type_data(cfg: DictConfig):
        inter_td = InteractionGraph.get_type_data(cfg)
        is_act_td = TensorTD((1,), dtype=bool)
        return ClassTD(InteractionActivityData, graph=inter_td, is_active=is_act_td)

class IsActiveData(StructData):
    lig: MolGraph
    rec: ProtGraph
    is_active: torch.Tensor

    @staticmethod
    def get_type_data(cfg: DictConfig):
        lig_td = MolGraph.get_type_data(cfg)
        rec_td = ProtGraph.get_type_data(cfg)
        act_td = TensorTD((1,), dtype=bool)
        return ClassTD(ActivityData, lig=lig_td, rec=rec_td, is_active=act_td)

class IsActiveIndexData(StructData):
    lig: MolGraph
    rec: ProtGraph
    is_active: torch.Tensor
    index: int

    @staticmethod
    def get_type_data(cfg: DictConfig):
        lig_td = MolGraph.get_type_data(cfg)
        rec_td = ProtGraph.get_type_data(cfg)
        act_td = TensorTD((1,), dtype=bool)
        index_td = TensorTD((1,), dtype=int)
        return ClassTD(ActivityData, lig=lig_td, rec=rec_td, is_active=act_td, index=index_td)

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
