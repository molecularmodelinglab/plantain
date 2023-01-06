from typing import Callable, Optional, Set, Type
from terrace import collate
from data_formats.base_formats import Data, Input, Prediction
from data_formats.tasks import Task

class Model():   
    
    def predict(self, tasks: Set[Type[Task]], x: Input) -> Prediction:
        ret = []
        my_tasks = self.get_tasks()
        for task in tasks:
            assert task in my_tasks
            method_name = task.get_name()
            single_method_name = task.get_name() + "_single"
            if hasattr(self, method_name):
                ret.append(getattr(self, method_name)(x))
            elif hasattr(self, single_method_name):
                pred = collate([ getattr(self, single_method_name)(item) for item in x ])
                ret.append(pred)
            else:
                raise AttributeError(f"Do perform {task.get_name()}, model must have either {method_name} or {single_method_name} methods")
        # smh there's a terrace bug...
        return ret[0] # Data.merge(ret)

    @staticmethod
    def get_name() -> str:
        raise NotImplementedError()

    def get_tasks(self) -> Set[Type[Task]]:
        raise NotImplementedError()

    def get_data_format(self) -> Optional[Callable[[Input], Input]]:
        raise NotImplementedError()