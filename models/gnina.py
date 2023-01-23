from data_formats.base_formats import Data
from data_formats.tasks import ScoreActivityClass, ScoreActivityRegr, ScorePose
from datasets.bigbind_gnina import LigAndRecGnina
from models.model import ScoreActivityModel
from terrace.batch import Batch

class Gnina(ScoreActivityModel):

    def __init__(self, conf_id: int):
        self.conf_id = conf_id

    def call_single(self, x: LigAndRecGnina):
        return x.affinities[self.conf_id], x.pose_scores[self.conf_id]

    def get_data_format(self):
        return None

    @staticmethod
    def get_name() -> str:
        return "gnina"

    def get_tasks(self):
        return [ ScoreActivityRegr, ScoreActivityClass, ScorePose ]

    def predict(self, tasks, x):
        affinity, pose = self(x)
        all_pose_scores = x.pose_scores
        p1 = super().make_prediction(affinity)
        p2 = Batch(ScorePose.Prediction, pose_scores=all_pose_scores)
        return Data.merge([p1, p2])
