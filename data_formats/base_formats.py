import typing
import torch
from inspect import Signature, Parameter
from dataclasses import dataclass
from rdkit import Chem
from Bio.PDB.Model import Model
from typing import List, Optional, Tuple, Type, TypeVar, Generic
from terrace import Batchable

class Data(Batchable):
    """ Base class for all the input, label, and prediction formats. Main 
    thing is that we can dynamically create objects that inherit from several
    classes (e.g. activity and is_active data) at runtime"""
    
    def create(subclasses: Tuple[Type["Data"]], **kwargs):
        """creates a new datapoint that inherits from all subclasses,
        whose members are defined by the kwargs"""

        type_name = "Data[" + ", ".join(map(lambda cls: cls.__qualname__, subclasses)) + "]"
        
        # Raise a proper TypeError if the user doesn't give the correct arguments
        # (that is, all the members of all the subclasses)
        type_keys = set().union(*[set(typing.get_type_hints(cls).keys()) for cls in subclasses ])
        s = Signature([Parameter(name, Parameter.KEYWORD_ONLY) for name in type_keys])
        s.bind(**kwargs)

        return dataclass(type(type_name, subclasses, {}))(**kwargs)

    def merge(items: List["Data"]):
        """ Merge items of different data subclasses into a single object """
        subclasses = tuple(map(type, items))
        kwargs = {}
        for item in items:
            kwargs.update(item.__dict__)
        return Data.create(subclasses, **kwargs)

class Input(Data):
    pass

class Label(Data):
    pass

class Prediction(Data):
    pass

@dataclass
class LigAndRec(Input):
    lig: Chem.Mol
    rec: Model

@dataclass
class LigAndRecDocked(LigAndRec):
    # The number of docked poses for each datapoint is not necessarily
    # the same, so we must define a custom collate method
    docked_scores: torch.Tensor

    @staticmethod
    def collate_docked_scores(all_docked_scores: List[torch.Tensor]) -> List[torch.Tensor]:
        return all_docked_scores

@dataclass
class IsActive(Label):
    is_active: bool

@dataclass
class Activity(Label):
    activity: bool

