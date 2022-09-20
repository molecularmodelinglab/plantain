from typing import List, Tuple, Optional
import torch
import numpy as np
import scipy.spatial as spa

from terrace.batch import Batch
from datasets.graphs.graph3d import Graph3d, Node3d, Edge3d

def make_knn_edgelist(nodes: List[Node3d],
                      radius: float,
                      max_neighbors: Optional[int]) -> List[Tuple[int, int]]:
    """ Returns the knn edgelist """
    coords = Batch(nodes).coord
    num_nodes = coords.shape[0]
    distance = spa.distance.cdist(coords, coords)
    ret = set()
    for i in range(num_nodes):
        dst = list(np.where(distance[i, :] < radius)[0])
        dst.remove(i)
        if max_neighbors != None and len(dst) > max_neighbors:
            dst = list(np.argsort(distance[i, :]))[1: max_neighbors + 1]  # closest would be self loop
        if len(dst) == 0:
            dst = list(np.argsort(distance[i, :]))[1:2]  # closest would be the index i itself > self loop
        if i in dst:
            dst.remove(i)
        assert i not in dst
        for d in dst:
            ret.add(frozenset([i, d]))
    return [ tuple(edge) for edge in ret ]
