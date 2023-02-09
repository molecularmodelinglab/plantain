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
        confs = LigAndRecGraphMultiPose.get_conformers(cfg, data.lig)
        lig_graphs = tuple([ MolGraph(cfg, data.lig, c) for c in confs])
        rec_graph = ProtGraph(cfg, data.rec)
        return LigAndRecGraphMultiPose(lig_graphs, rec_graph)

    # @staticmethod
    # def collate_lig_graphs(x):
    #     return x

    @staticmethod
    def get_conformers(cfg, lig):
        sample = cfg.data.pose_sample
        n_confs = lig.GetNumConformers()
        num_poses = cfg.data.get("num_poses", None)
        if sample == 'all':
            assert num_poses is None
            return range(n_confs)
        elif sample == 'best_and_worst':
            n_poses = 2 if num_poses is None else num_poses
            ret = []
            for n in range(n_poses-1):
                ret.append(min(n,n_confs))
            return ret + [n_confs - 1]
        elif sample == 'worst_and_best':
            assert num_poses is None
            return [n_confs - 1, 0]