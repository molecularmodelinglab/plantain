from data_formats.base_formats import ActivityData, ActivityPassThrough

class LigRecGraphData(ActivityPassThrough):

    lig: MolGraph
    rec: ProtGraph


    def __init__(self, act_data: ActivityData):
        super().__init__(act_data)
