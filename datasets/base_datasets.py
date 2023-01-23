from typing import Set, Tuple, Type
from torch.utils import data
from data_formats.base_formats import Data, Input, Label
from data_formats.tasks import Task

class Dataset(data.Dataset):

    def __init__(self, cfg, transform=None):
        super().__init__()
        self.cfg = cfg
        self.transform = transform
        
    @staticmethod
    def get_name() -> str:
        raise NotImplementedError()

    def get_label_classes(self) -> Set[Type[Task]]:
        raise NotImplementedError()

    def get_tasks(self) -> Set[Task]:
        """ Find all tasks this dataset can be used for. Looks for all
        tasks whose Label class cooresponds to one of our label types"""
        ret = set()
        label_classes = self.get_label_classes()
        for task in Task.__subclasses__():
            if task.Label in label_classes:
                ret.add(task)
        return ret

    def len_impl(self):
        raise NotImplementedError

    def __len__(self):
        if "debug_dataset_len" in self.cfg and self.cfg.debug_dataset_len is not None:
            return self.cfg.debug_dataset_len
        return self.len_impl()

    def getitem_impl(self, index: int) -> Tuple[Input, Label]:
        raise NotImplementedError

    def __getitem__(self, index: int) -> Tuple[Input, Label]:

        if index >= len(self):
            raise IndexError()

        try:
            ret = self.getitem_impl(index)
            if self.transform is not None:
                x, y = ret
                # we want all the attributes of the og data + the processed data
                x_trans = self.transform(self.cfg, x)
                return Data.merge([x, x_trans]), y
            else:
                return ret
        except KeyboardInterrupt:
            raise
        except:
            print(f"Error proccessing item at {index=}")
            raise