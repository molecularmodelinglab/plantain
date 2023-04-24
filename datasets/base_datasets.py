from typing import List, Set, Tuple, Type
from torch.utils import data
from data_formats.tasks import Task
from data_formats.transforms import Transform
from terrace.dataframe import DFRow

class Dataset(data.Dataset):

    def __init__(self, cfg, x_transforms=[]):
        super().__init__()
        self.cfg = cfg
        self.x_transforms = x_transforms.copy()
        if "remove_feats" in self.cfg.data and self.get_name() in self.cfg.data.remove_feats:
            for feat in self.cfg.data.remove_feats[self.get_name()]:
                if feat in self.x_transforms:
                    self.x_transforms.remove(feat)
        self.required_x_features = Transform.get_required_features(self.x_transforms)
        
    @staticmethod
    def get_name() -> str:
        raise NotImplementedError()

    def get_label_feats(self) -> List[str]:
        raise NotImplementedError

    def get_tasks(self) -> List[str]:
        """ Find all tasks this dataset can be used for. Looks for all
        tasks whose Label class cooresponds to one of our label types"""
        ret = []
        label_feats = self.get_label_feats()
        for name, task in Task.all_tasks.items():
            if set(task.label_feats).issubset(set(label_feats)):
                ret.append(name)
        return ret

    def len_impl(self):
        raise NotImplementedError

    def __len__(self):
        if "debug_dataset_len" in self.cfg and self.cfg.debug_dataset_len is not None:
            return self.cfg.debug_dataset_len
        return self.len_impl()

    def getitem_impl(self, index: int) -> Tuple[DFRow, DFRow]:
        raise NotImplementedError

    def __getitem__(self, index: int) -> Tuple[DFRow, DFRow]:

        if index >= len(self):
            raise IndexError()

        try:
            x, y = self.getitem_impl(index)
            # todo: this whole "remove_feats" thing smells like I'm using
            # a wrong abstraction for this
            return Transform.apply_many(self.cfg, self.x_transforms, x), y
        except KeyboardInterrupt:
            raise
        except:
            print(f"Error proccessing item at {index}")
            raise