import abc
from terrace.batch import Batchable

class Task:

    class Data(Batchable):
        pass

    @staticmethod
    @abc.abstractmethod
    def get_metrics():
        pass

    # todo: this belongs in the model base class
    @classmethod
    def get_prediction(cls, model, x):
        method_name = "predict_" + cls.__name__
        return getattr(model, method_name)(x)

class ActivityClassification(Task):

    class Data(Task.Data):
        is_active: float

    @staticmethod
    def get_metrics():
        pass