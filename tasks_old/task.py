from data_formats.base_formats import RawData

class Task:

    @staticmethod
    def get_metrics():
        raise NotImplementedError()

    @staticmethod
    def get_name() -> str:
        raise NotImplementedError()