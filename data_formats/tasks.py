from dataclasses import dataclass
from typing import List
from data_formats.base_formats import Activity, LigAndRec, Label, Prediction, IsActive

class Task:
    """ Note: _all_ tasks should be defined in this file. This is required
    for Datasets to know which tasks they qualify for (they loop over all
    the subclasses of Task) """
    @staticmethod
    def get_name() -> str:
        raise NotImplementedError()

class ScoreActivityClass(Task):

    Input = LigAndRec
    Label = IsActive

    class Prediction(Prediction):
        is_active_score: float
        
    @staticmethod
    def get_name() -> str:
        return "score_activity_class"

class ScoreActivityRegr(Task):

    Input = LigAndRec
    Label = Activity

    class Prediction(Prediction):
        activity_score: float
        
    @staticmethod
    def get_name() -> str:
        return "score_activity_regr"

class ClassifyActivity(Task):

    Input = LigAndRec
    Label = IsActive

    class Prediction(Prediction):
        active_prob_unnorm: float
        active_prob: float

    @staticmethod
    def get_name() -> str:
        return "classify_activity"

class ScorePose(Task):

    Input = LigAndRec

    class Label(Label):
        pose_rmsds: List[float]

    class Prediction(Prediction):
        pose_scores: List[float]

    @staticmethod
    def get_name() -> str:
        return "score_pose"