from typing import Callable, Optional, Set, Type
from data_formats.base_formats import Data, Input, Prediction
from data_formats.tasks import Task

class Model():   
    
    def predict(self, tasks: Set[Type[Task]], x: Input) -> Prediction:
        ret = []
        my_tasks = self.get_tasks()
        for task in tasks:
            assert task in my_tasks
            method_name = task.get_name()
            ret.append(getattr(self, method_name)(x))
        # todo: merge everythig once we get merged batches working
        return ret[0] # Data.merge(ret)

    @staticmethod
    def get_name() -> str:
        raise NotImplementedError()

    def get_tasks(self) -> Set[Type[Task]]:
        raise NotImplementedError()

    def get_data_format(self) -> Optional[Callable[[Input], Input]]:
        raise NotImplementedError()