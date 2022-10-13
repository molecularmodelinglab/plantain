from torch.utils.data import Sampler, RandomSampler
from tqdm import trange
import os
import pandas as pd

from datasets.graphs.prot_graph import get_node_and_edge_nums_from_pdb
from datasets.graphs.mol_graph import get_node_and_edge_nums_from_sdf
from datasets.cacheable_dataset import get_dataset_hash

class EdgeNumSampler(Sampler):

    def __init__(self, cfg, dataset):
        super().__init__(dataset)
        self.dataset = dataset
        self.max_rec_edges = cfg.sampler.max_rec_edges
        self.max_lig_edges = cfg.sampler.max_lig_edges
        self.graph_size_df = self.get_all_edge_and_node_nums(cfg)

        self.samples = []

        inner_sampler = RandomSampler(dataset)
        cur_rec_edges = 0
        cur_lig_edges = 0
        cur_sample = []

        for idx in inner_sampler:
            if idx > 1000: continue
            rec_edges = self.graph_size_df.rec_edges[idx]
            lig_edges = self.graph_size_df.lig_edges[idx]
            if rec_edges > self.max_rec_edges or lig_edges > self.max_lig_edges:
                print(f"Skipping {idx} because there are too many edges")
            
            cur_lig_edges += lig_edges
            cur_rec_edges += rec_edges
            if cur_rec_edges > self.max_rec_edges or cur_lig_edges > self.max_lig_edges:
                self.samples.append(cur_sample)
                cur_rec_edges = rec_edges
                cur_lig_edges = lig_edges
                cur_sample = []

            cur_sample.append(idx)

    def __iter__(self):
        for sample in self.samples:
            yield sample
        
    def get_all_edge_and_node_nums(self, cfg, cache=True):

        cache_fname = cfg.platform.cache_dir + "/" + cfg.dataset + "/" + get_dataset_hash(cfg) + "/graph_sizes.csv"
        if cache:
            try:
                return pd.read_csv(cache_fname)
            except FileNotFoundError:
                pass

        lig_nodes = []
        lig_edges = []
        rec_nodes = []
        rec_edges = []
        for idx in trange(len(self.dataset)):
            rec_file = self.dataset.get_rec_file(idx)
            lig_file = self.dataset.get_lig_file(idx)
            rn, re = get_node_and_edge_nums_from_pdb(cfg, rec_file)
            ln, le = get_node_and_edge_nums_from_sdf(cfg, lig_file)
            lig_nodes.append(ln)
            lig_edges.append(le)
            rec_nodes.append(rn)
            rec_edges.append(re)
            # todo: remove
            if idx > 10000:
                break

        ret = pd.DataFrame({
            "lig_nodes": lig_nodes,
            "lig_edges": lig_edges,
            "rec_nodes": rec_nodes,
            "rec_edges": rec_edges
        })
        cache_folder = "/".join(cache_fname.split("/")[:-1])
        os.makedirs(cache_folder, exist_ok=True)
        ret.to_csv(cache_fname, index=False)
        return ret