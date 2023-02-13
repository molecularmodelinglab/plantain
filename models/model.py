from typing import Callable, List, Optional, Set, Type
import torch
from data_formats.tasks import Task
from terrace import collate, Batch
from terrace.dataframe import DFRow, merge

class Model():
    
    def predict(self, x: Batch[DFRow], task_names: Optional[List[str]] = None) -> Batch[DFRow]:
        if task_names is None:
            task_names = self.get_tasks()
        pred = self(x)
        ret = [ pred ]
        if len(task_names) == 0:
            raise ValueError("predict needs at least one task")
        for task_name in task_names:
            task = Task.all_tasks[task_name]
            method_name = task_name
            if hasattr(self, method_name):
                ret.append(getattr(self, method_name)(x, pred))
            else:
                all_feats = pred.attribute_names()
                for feat in task.pred_feats:
                    if feat not in all_feats:
                        raise AttributeError(f"To perform {task_name}, model must have the {method_name} method (or the __call__ method should return a batch with {feat}")
        return merge(ret)

    def predict_train(self, x, y, task_names):
        return self.predict(x, task_names)

    @staticmethod
    def get_name() -> str:
        raise NotImplementedError()

    def get_tasks(self) -> List[str]:
        raise NotImplementedError()

    def get_input_feats(self) -> List[str]:
        return []

    def call_single(self, x):
        raise NotImplementedError

    def __call__(self, x):
        return collate([ self.call_single(item) for item in x ])

class ScoreActivityClassModel(Model):

    def get_tasks(self):
        return [ "score_activity_class" ]

    def score_activity_class(self, x, pred):
        return Batch(DFRow, is_active_score=pred.score)

class ScoreActivityRegrModel(Model):

    def get_tasks(self):
        return [ "score_activity_regr" ]

    def score_activity_regr(self, x, pred):
        return Batch(DFRow, activity_score=pred.score)

class ScoreActivityModel(ScoreActivityClassModel, ScoreActivityRegrModel):
    
    def get_tasks(self):
        return [ "score_activity_class", "score_activity_regr" ]

class ClassifyActivityModel(ScoreActivityClassModel):

    def get_tasks(self):
        return [ "score_activity_class", "classify_activity" ]

    def classify_activity(self, x, pred):
        unnorm = pred.score
        prob = torch.sigmoid(unnorm)
        return Batch(DFRow, active_prob_unnorm=unnorm, active_prob=prob)

# class RandomPoseScore(Model):

#     def get_tasks(self):
#         return [ ScorePose ]

#     def __call__(self, x):
#         return [ torch.randn((lig.GetNumConformers(),), dtype=torch.float32) for lig in x.lig ]

#     def score_pose(self, x, pred):
#         type_ = self.get_pred_type()
#         return Batch(type_, pose_scores=pred)