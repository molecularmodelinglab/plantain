from dataclasses import dataclass
from typing import List
from data_formats.base_formats import Activity, InvDistMat, LigAndRec, Label, PoseRMSDs, Prediction, IsActive

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

    Label = PoseRMSDs

    class Prediction(Prediction):
        pose_scores: List[float]

    @staticmethod
    def get_name() -> str:
        return "score_pose"

class PredictInteractionMat(Task):

    Input = LigAndRec
    Label = InvDistMat
    Prediction = InvDistMat

    @staticmethod
    def get_name() -> str:
        return "predict_interaction_mat"

# todo: better name
class RejectOption(Task):

    Input = None
    Label = None

    class Prediction(Prediction):
        select_score: float

    @staticmethod
    def get_name() -> str:
        return "reject_option"
