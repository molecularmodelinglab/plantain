from typing import Callable, List, Optional, Set, Type
import torch
from terrace import collate, Batch
from data_formats.base_formats import Data, Input, Prediction
from data_formats.tasks import ClassifyActivity, ScoreActivityClass, ScoreActivityRegr, ScorePose, Task

class Model():   
    
    def predict(self, tasks: Set[Type[Task]], x: Input) -> Prediction:
        ret = []
        my_tasks = self.get_tasks()
        pred = self(x)
        # if hasattr(self, "__call__"):
        #     pred = self(x)
        # elif hasattr(self, "call_single"):
        #     pred = collate([ self.call_single(item) for item in x ])
        # else:
        #     raise AttributeError("Model must implement either a __call__ or call_single method")
        if len(tasks) == 0:
            raise ValueError("predict needs at least one task")
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

    def get_tasks(self) -> List[Type[Task]]:
        raise NotImplementedError()

    def get_pred_type(self) -> Type[Prediction]:
        return Data.create_type([ t.Prediction for t in self.get_tasks() ])

    def get_data_format(self) -> Optional[Callable[[Input], Input]]:
        return None

    def call_single(self, x):
        raise NotImplementedError

    def __call__(self, x):
        return collate([ self.call_single(item) for item in x ])

class ScoreActivityClassModel(Model):

    def get_tasks(self):
        return [ ScoreActivityClass ]

    def score_activity_class(self, x, pred):
        return Batch(ScoreActivityClass.Prediction, is_active_score=pred)

class ScoreActivityRegrModel(Model):

    def get_tasks(self):
        return [ ScoreActivityRegr ]

    def score_activity_regr(self, x, pred):
        return Batch(ScoreActivityRegr.Prediction, activity_score=pred)

class ScoreActivityModel(ScoreActivityClassModel, ScoreActivityRegrModel):
    
    def get_tasks(self):
        return [ ScoreActivityRegr, ScoreActivityClass ]

    def make_prediction(self, score):
        type_ = self.get_pred_type()
        return Batch(type_, activity_score=score, is_active_score=score)

class ClassifyActivityModel(ScoreActivityClassModel):

    def get_tasks(self):
        return [ ScoreActivityClass, ClassifyActivity ]

    def classify_activity(self, x, pred):
        unnorm = pred
        prob = torch.sigmoid(unnorm)
        return Batch(ClassifyActivity.Prediction, active_prob_unnorm=unnorm, active_prob=prob)

    def make_prediction(self, unnorm):
        prob = torch.sigmoid(unnorm)
        type_ = Data.create_type([ScoreActivityClass, ClassifyActivity]) # self.get_pred_type()
        return Batch(type_, active_prob_unnorm=unnorm, active_prob=prob, is_active_score=unnorm)

class RandomPoseScore(Model):

    def get_tasks(self):
        return [ ScorePose ]

    def __call__(self, x):
        return [ torch.randn((lig.GetNumConformers(),), dtype=torch.float32) for lig in x.lig ]

    def score_pose(self, x, pred):
        type_ = self.get_pred_type()
        return Batch(type_, pose_scores=pred)