from dataclasses import dataclass
from tasks.task import Task

class ActivityClassification(Task):

    @staticmethod
    def get_metrics():
        pass

    @staticmethod
    def get_name():
        return "activity_classification"