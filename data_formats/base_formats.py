import typing
import torch
from inspect import Signature, Parameter
from rdkit import Chem
from Bio.PDB.Model import Model
from typing import List, Optional, Tuple, Type, TypeVar, Generic, get_type_hints
from terrace import Batchable, Batch


class Data(Batchable):
    """ Base class for all the input, label, and prediction formats. Main 
    thing is that we can dynamically create objects that inherit from several
    classes (e.g. activity and is_active data) at runtime"""
    
    @staticmethod
    def create_type(subclasses: Tuple[Type["Data"]]):

        type_name = "Data[" + ", ".join(map(lambda cls: cls.__qualname__, subclasses)) + "]"
        if type_name in globals():
            type_ = globals()[type_name]
        else:
            type_ = type(type_name, tuple(subclasses), {})
            globals()[type_name] = type_
        type_.__module__ = __name__

        return type_

    @staticmethod
    def create(subclasses: Tuple[Type["Data"]], **kwargs):
        """creates a new datapoint that inherits from all subclasses,
        whose members are defined by the kwargs"""
        
        # Raise a proper TypeError if the user doesn't give the correct arguments
        # (that is, all the members of all the subclasses)
        type_keys = set().union(*[set(typing.get_type_hints(cls).keys()) for cls in subclasses ])
        s = Signature([Parameter(name, Parameter.KEYWORD_ONLY) for name in type_keys])
        s.bind(**kwargs)
        
        type_ = Data.create_type(subclasses)

        return type_(**kwargs)

    @staticmethod
    def merge(items: List["Data"]):
        """ Merge items of different data subclasses into a single object """
        assert len(items) > 0

        if isinstance(items[0], Batch):
            """ Very hacky, relies on implimentation details of terrace. Will add support
            for this kind of thing in terrace shortly"""
            batch_subclasses = tuple(map(lambda item: item.item_type(), items))
            # type_name = "Data[" + ", ".join(map(lambda cls: cls.__qualname__, batch_subclasses)) + "]"
            # if type_name in globals():
            #     batch_subclass = globals()[type_name]
            # else:
            #     batch_subclass = type(type_name, batch_subclasses, {})
            #     batch_subclass.__module__ = __name__

            batch_subclass = Data.create_type(batch_subclasses)
                
            kwargs = {}
            for item in items:
                kwargs.update(item.asdict())

            return Batch(batch_subclass, **kwargs)
        else:
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

class LigAndRec(Input):
    lig: Chem.Mol
    rec: Model
    pocket_id: str

class LigAndRecDocked(LigAndRec):
    # The number of docked poses for each datapoint is not necessarily
    # the same, so we must define a custom collate method
    docked_scores: torch.Tensor

    @staticmethod
    def collate_docked_scores(all_docked_scores: List[torch.Tensor]) -> List[torch.Tensor]:
        return all_docked_scores

class IsActive(Label):
    is_active: bool

class Activity(Label):
    activity: bool

class Pose(Label):
    lig_coords: torch.Tensor

    @staticmethod
    def collate_lig_coords(all_lig_coords: List[torch.Tensor]) -> List[torch.Tensor]:
        return all_lig_coords

    # @staticmethod
    # def collate_rec_coords(x):
    #     return x

class InvDistMat(Prediction):
    inv_dist_mat: torch.Tensor

    @staticmethod
    def collate_inv_dist_mat(x):
        return x


