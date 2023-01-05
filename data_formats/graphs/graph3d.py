from typing import List, Tuple, Optional
import torch

from terrace.graph import Graph
from terrace.batch import Batchable

class Node3d(Batchable):
    coord: torch.Tensor # ((num_nodes, 3))
    cat_feat: torch.Tensor # ((num_nodes, node_cat_feat), dtype=torch.long)
    scal_feat: torch.Tensor # ((num_nodes, node_scal_feat))

    # these are plotting properties, only update if you want your
    # plotly plots of these graphs to be prettier

    def get_color(self) -> str:
        return "#ff0000"

    def get_radius(self) -> float:
        return 0.5

class Edge3d(Batchable):
    cat_feat: torch.Tensor # ((num_edges, edge_cat_feat), dtype=torch.long)
    scal_feat: torch.Tensor # ((num_edges, edge_scal_feat))

    def get_color(self) -> str:
        return "#000000"

class Graph3d(Graph[Node3d, Edge3d]):
    def __init__(self, nodes: List[Node3d],
                 edges: List[Tuple[int, int]],
                 edata: List[Edge3d],
                 directed: bool = False):
        super(Graph3d, self).__init__(nodes, edges, edata, directed)