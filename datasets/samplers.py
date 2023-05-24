
import random


class LigTimesRecSampler:
    """ The 'size' of a batch is computed as B*max(L*R), 
    where B is the batch size, and max(L*R) is the maximum
    of a value that's supposed to correlate with the number
    of ligand nodes times the number of rec pocket nodes. This
    sampler greedily adds indexes to the current batch until
    the batch 'size' reaches cfg.batch_sampler.max_size """

    def __init__(self, cfg, dataset, shuffle):
        self.cfg = cfg
        self.dataset = dataset
        self.shuffle = shuffle

    def __iter__(self):
        sample_sizes = self.dataset.structures.lig_smiles.str.len()*self.dataset.structures.crossdock_num_pocket_residues
        cur_batch = []
        max_size = 0

        idxs = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(idxs)

        for i, idx in enumerate(idxs):
            next_size = sample_sizes[idx]
            next_batch_size = max(max_size, next_size)*(len(cur_batch) + 1)
            if next_batch_size < self.cfg.batch_sampler.max_size:
                cur_batch.append(idx)
                max_size = max(max_size, next_size)
            else:
                if len(cur_batch) == 0:
                    # print(f"Skipping item at index {idx} because it is singlehandedly too big {next_batch_size=}")
                    print(f"WARNING: item at index {idx} is singlehandedly too big {next_batch_size=}.")
                else:
                    if self.cfg.batch_sampler.debug:
                        print(f"Sampling {cur_batch} size={max_size*len(cur_batch)}")
                    yield cur_batch
                cur_batch = [ idx ]
                max_size = next_size

        yield cur_batch

class RecSampler:
    """ Like above, but size is just the rec nodes """

    def __init__(self, cfg, dataset, shuffle):
        self.cfg = cfg
        self.dataset = dataset
        self.shuffle = shuffle

    def __iter__(self):
        sample_sizes = self.dataset.structures.crossdock_num_pocket_residues
        cur_batch = []
        max_size = 0

        idxs = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(idxs)

        for i, idx in enumerate(idxs):
            next_size = sample_sizes[idx]
            next_batch_size = max(max_size, next_size)*(len(cur_batch) + 1)
            if next_batch_size < self.cfg.batch_sampler.max_size:
                cur_batch.append(idx)
                max_size = max(max_size, next_size)
            else:
                if len(cur_batch) == 0:
                    # print(f"Skipping item at index {idx} because it is singlehandedly too big {next_batch_size=}")
                    print(f"WARNING: item at index {idx} is singlehandedly too big {next_batch_size=}.")
                else:
                    if self.cfg.batch_sampler.debug:
                        print(f"Sampling {cur_batch} size={max_size*len(cur_batch)}")
                    yield cur_batch
                cur_batch = [ idx ]
                max_size = next_size

        yield cur_batch

SAMPLERS= {
    "lig_times_rec": LigTimesRecSampler,
    "rec": RecSampler,
}