from torch.utils import data

class ComboDataloader(data.DataLoader):

    def __init__(self, loaders):
        self.loaders = loaders
        self.num_workers = loaders[0].num_workers
        self.collate_fn = loaders[0].collate_fn

    def get_dataset_index(self, batch_idx):
        return batch_idx % len(self.loaders)

    def __len__(self) -> int:
        return max([len(d) for d in self.loaders])

    def __iter__(self):
        stopped = [ False for loader in self.loaders ]
        iterators = [ iter(loader) for loader in self.loaders ]
        while True:
            for i in range(len(iterators)):
                try:
                    batch = next(iterators[i])
                except StopIteration:
                    stopped[i] = True
                    if sum(stopped) == len(stopped):
                        return
                    iterators[i] = iter(self.loaders[i])
                    batch = next(iterators[i])
                yield batch