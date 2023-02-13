from models.model import ScoreActivityModel
from terrace.batch import Batch
from terrace.dataframe import DFRow

class Gnina(ScoreActivityModel):

    def __init__(self, conf_id: int):
        self.conf_id = conf_id

    def call_single(self, x: DFRow):
        return DFRow(score=x.gnina_affinities[self.conf_id], pose_scores=x.gnina_pose_scores[self.conf_id])

    def get_input_feats(self):
        return ["gnina_affinities", "gnina_pose_scores"]

    @staticmethod
    def get_name() -> str:
        return "gnina"

    def get_tasks(self):
        return [ "score_activity_regr", "score_activity_class", "score_pose" ]
