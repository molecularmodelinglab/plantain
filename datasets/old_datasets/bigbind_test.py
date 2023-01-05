from datasets.bigbind_act import BigBindActDataset

class BigBindTestDataset(BigBindActDataset):

    def __init__(self, cfg, split):
        super().__init__(cfg, split)

    def __getitem__(self, index):
        return super().__getitem__(0)