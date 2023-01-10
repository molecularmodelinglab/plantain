from typing import Callable, Optional, Set, Type
import torch
from terrace import collate, Batch
from data_formats.base_formats import Data, Input, Prediction
from data_formats.tasks import ClassifyActivity, ScoreActivityClass, Task

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
                raise AttributeError(f"To perform {task.get_name()}, model must have either {method_name} or {single_method_name} methods")
        return Data.merge(ret)

    @staticmethod
    def get_name() -> str:
        raise NotImplementedError()

    def get_tasks(self) -> Set[Type[Task]]:
        raise NotImplementedError()

    def get_data_format(self) -> Optional[Callable[[Input], Input]]:
        raise NotImplementedError()

class ScoreActivityClassModel(Model):

    def get_tasks(self):
        return { ScoreActivityClass }

class ClassifyActivityModel(ScoreActivityClassModel):

    def get_tasks(self):
        return { ScoreActivityClass, ClassifyActivity }

    def score_activity_class(self, x):
        score = self.classify_activity(x).active_prob_unnorm
        return Batch(ScoreActivityClass.Prediction, is_active_score=score)

    def classify_activity(self, x):
        unnorm = self(x)
        prob = torch.sigmoid(unnorm)
        return Batch(ClassifyActivity.Prediction, active_prob_unnorm=unnorm, active_prob=prob)