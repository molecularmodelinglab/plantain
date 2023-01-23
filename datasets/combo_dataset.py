from torch.utils import data

class ComboDataset(data.Dataset):

    def __init__(self, datasets):
        self.datasets = datasets

    def __len__(self):
        return max(map(len, self.datasets))

    def __getitem__(self, index: int):

        if index >= len(self):
            raise IndexError()

        ret = []
        for dataset in self.datasets:
            ret.append(dataset[index % len(dataset)])

        return tuple(ret)