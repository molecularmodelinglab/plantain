from data_formats.base_formats import LigAndRecDocked
from models.model import ScoreActivityModel

class Vina(ScoreActivityModel):

    def __init__(self, conf_id: int):
        self.conf_id = conf_id

    def call_single(self, x: LigAndRecDocked):
        return x.docked_scores[self.conf_id]

    def get_data_format(self):
        return None

    @staticmethod
    def get_name() -> str:
        return "vina"