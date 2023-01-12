from data_formats.base_formats import Input
from .mol_graph import MolGraph
from .prot_graph import ProtGraph

class LigAndRecGraph(Input):

    lig_graph: MolGraph
    rec_graph: ProtGraph

    @staticmethod
    def make(cfg, data):
        lig_graph = MolGraph(cfg, data.lig)
        rec_graph = ProtGraph(cfg, data.rec)
        return LigAndRecGraph(lig_graph, rec_graph)