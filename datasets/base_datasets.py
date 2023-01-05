from typing import Set, Tuple, Type
from torch.utils import data
from data_formats.base_formats import Input, Label
from data_formats.tasks import Task

class Dataset(data.Dataset):

    def __init__(self, transform=None):
        super().__init__()
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

    def getitem_impl(self, index: int) -> Tuple[Input, Label]:
        raise NotImplementedError

    def __getitem__(self, index: int) -> Tuple[Input, Label]:

        if index >= len(self):
            raise IndexError()

        try:
            ret = self.getitem_impl(index)
            if self.transform is not None:
                return self.transform(ret)
            else:
                return ret
        except KeyboardInterrupt:
            raise
        except:
            print(f"Error proccessing item at {index=}")
            raise