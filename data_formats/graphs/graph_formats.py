from typing import List, Tuple
from data_formats.base_formats import Input
from .mol_graph import MolGraph
from .prot_graph import ProtGraph

class LigAndRecGraph(Input):

    lig_graph: MolGraph
    rec_graph: ProtGraph

    @staticmethod
    def make(cfg, data):
        # this is very hacky
        conf = 0 if data.lig.GetNumConformers() > 0 else None
        lig_graph = MolGraph(cfg, data.lig, None)
        rec_graph = ProtGraph(cfg, data.rec)
        return LigAndRecGraph(lig_graph, rec_graph)

class LigAndRecGraphMultiPose(Input):

    lig_graphs: Tuple[MolGraph, ...]
    rec_graph: ProtGraph

    @staticmethod
    def make(cfg, data):
        lig_graphs = tuple([ MolGraph(cfg, data.lig, c) for c in LigAndRecGraphMultiPose.get_conformers(cfg, data.lig)])
        rec_graph = ProtGraph(cfg, data.rec)
        return LigAndRecGraphMultiPose(lig_graphs, rec_graph)

    # @staticmethod
    # def collate_lig_graphs(x):
    #     return x

    @staticmethod
    def get_conformers(cfg, lig):
        sample = cfg.data.pose_sample
        n_confs = lig.GetNumConformers()
        if sample == 'all':
            return range(n_confs)
        elif sample == 'best_and_worst':
            return [0, n_confs - 1]
        elif sample == 'worst_and_best':
            return [n_confs - 1, 0]