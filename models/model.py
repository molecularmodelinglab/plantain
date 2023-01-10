from typing import Callable, Optional, Set, Type
import torch
from terrace import collate, Batch
from data_formats.base_formats import Data, Input, Prediction
from data_formats.tasks import ClassifyActivity, ScoreActivityClass, ScoreActivityRegr, Task

class Model():   
    
    def predict(self, tasks: Set[Type[Task]], x: Input) -> Prediction:
        ret = []
        my_tasks = self.get_tasks()
        if hasattr(self, "__call__"):
            pred = self(x)
        elif hasattr(self, "call_single"):
            pred = collate([ self.call_single(item) for item in x ])
        else:
            raise AttributeError("Model must implement either a __call__ or call_single method")
        for task in tasks:
            assert task in my_tasks
            method_name = task.get_name()
            if hasattr(self, method_name):
                ret.append(getattr(self, method_name)(x, pred))
            else:
                raise AttributeError(f"To perform {task.get_name()}, model must have the {method_name} method")
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

    def score_activity_class(self, x, pred):
        return Batch(ScoreActivityClass.Prediction, is_active_score=pred)

class ScoreActivityRegrModel(Model):

    def get_tasks(self):
        return { ScoreActivityRegr }

    def score_activity_regr(self, x, pred):
        return Batch(ScoreActivityRegr.Prediction, activity_score=pred)

class ScoreActivityModel(ScoreActivityClassModel, ScoreActivityRegrModel):
    
    def get_tasks(self):
        return { ScoreActivityRegr, ScoreActivityClass }

class ClassifyActivityModel(ScoreActivityClassModel):

    def get_tasks(self):
        return { ScoreActivityClass, ClassifyActivity }

    def classify_activity(self, x, pred):
        unnorm = pred
        prob = torch.sigmoid(unnorm)
        return Batch(ClassifyActivity.Prediction, active_prob_unnorm=unnorm, active_prob=prob)