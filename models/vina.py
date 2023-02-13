from models.model import ScoreActivityModel
from terrace.dataframe import DFRow

class Vina(ScoreActivityModel):

    def __init__(self, conf_id: int):
        self.conf_id = conf_id

    def call_single(self, x: DFRow):
        return x.docked_scores[self.conf_id]

    def get_data_format(self):
        return None

    @staticmethod
    def get_name() -> str:
        return "vina"