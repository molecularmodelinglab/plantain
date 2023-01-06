from dataclasses import dataclass

from data_formats.base_formats import Input
from .mol_graph import MolGraph
from .prot_graph import ProtGraph

@dataclass
class LigAndRecGraph(Input):

    lig_graph: MolGraph
    rec_graph: ProtGraph

    def __init__(self, cfg, data):
        self.lig_graph = MolGraph(cfg, data.lig)
        self.rec_graph = ProtGraph(cfg, data.rec)