from typing import List, Tuple, Optional
import torch
import numpy as np
import scipy.spatial as spa

from terrace.batch import Batch
from .graph3d import Graph3d, Node3d, Edge3d

def make_knn_edgelist(nodes: List[Node3d],
                      radius: float,
                      max_neighbors: Optional[int]):
    """ Returns the knn edgelist
    Note: an earlier version of this function returned only one permutation
    of (i,j) if (i,j) is an edge. Now this returns both permutations (i,j)
    and (j,i) """
    if isinstance(nodes, Batch):
        coords = nodes.coord
    else:
        coords = Batch(nodes).coord
    num_nodes = coords.shape[0]
    distance = spa.distance.cdist(coords, coords)
    ret = []
    seen = set()
    for i in range(num_nodes):
        dst = list(np.where(distance[i, :] < radius)[0])
        dst.remove(i)
        if max_neighbors is not None and len(dst) > max_neighbors:
            dst = list(np.argsort(distance[i, :]))[1: max_neighbors + 1]  # closest would be self loop
        if len(dst) == 0:
            dst = list(np.argsort(distance[i, :]))[1:2]  # closest would be the index i itself > self loop
        if i in dst:
            dst.remove(i)
        assert i not in dst
        for d in dst:
            if (i,d) not in seen:
                ret.append((i,d))
                ret.append((d,i))
                seen.add((i,d))
                seen.add((d,i))
    return ret, [ distance[i,j] for i,j in ret ]

def make_res_knn_edgelist(nodes: List[Node3d],
                          res_indexes: List[int],
                          atom_radius: float,
                          res_radius: float):
    "Returns residue KNN edgelist (will document soon...)"
    if isinstance(nodes, Batch):
        coords = nodes.coord
    else:
        coords = Batch(nodes).coord

    distance = spa.distance.cdist(coords, coords)
    num_nodes = coords.shape[0]
    ret, _ = make_knn_edgelist(nodes, atom_radius, None)
    num_res = max(res_indexes)
    res_nodes = {}
    for i, r in enumerate(res_indexes):
        if r not in res_nodes:
            res_nodes[r] = []
        res_nodes[r].append(i)

    for i in res_nodes:
        for j in res_nodes:
            if j <= i: continue
            min_dist = None
            min_edge = None
            for n1 in res_nodes[i]:
                for n2 in res_nodes[j]:
                    dist = distance[n1,n2] # torch.linalg.norm(coords[n1] - coords[n2])
                    if dist > res_radius: continue
                    if min_dist is None or dist < min_dist:
                        min_dist = dist
                        min_edge = (n1, n2)
            if min_edge is not None:
                n1, n2 = min_edge
                ret.append((n1, n2))
                ret.append((n2, n1))

    return ret, [ distance[i,j] for i,j in ret ]
