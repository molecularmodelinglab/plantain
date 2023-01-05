from terrace import Batch, collate
from data_formats.base_formats import LigAndRecDocked
from data_formats.tasks import ScoreActivity
from models.model import Model

class Vina(Model):

    def __init__(self, conf_id: int):
        self.conf_id = conf_id

    def score_activity(self, batch: Batch[LigAndRecDocked]) -> Batch[ScoreActivity.Prediction]:
        return collate([ self.score_activity_single(x) for x in batch ])

    def score_activity_single(self, x: LigAndRecDocked) -> ScoreActivity.Prediction:
        return ScoreActivity.Prediction(x.docked_scores[self.conf_id])

    def get_tasks(self):
        return { ScoreActivity }

    def get_data_format(self):
        return None

    @staticmethod
    def get_name() -> str:
        return "vina"