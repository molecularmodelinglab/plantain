
import torch
from typing import List, Optional, Tuple
from omegaconf import DictConfig
from rdkit import Chem
from Bio.PDB.Model import Model

from terrace.batch import Batch, make_batch
from terrace.graph import GraphTD
from terrace.type_data import ClassTD, TensorTD, ShapeVar

from datasets.graphs.graph3d import Graph3d, Node3d, Edge3d
from datasets.graphs.knn import make_knn_edgelist
from datasets.graphs.mol_graph import MolGraph, AtomNode, BondEdge
from datasets.graphs.prot_graph import ProtAtomNode, ProtResidueNode, ProtGraph
from datasets.graphs.dist_edge import DistEdge

class InteractionNode(Node3d):

    @staticmethod
    def get_type_data(cfg: DictConfig):
        assert cfg.data.rec_graph.node_type == "residue"
        atom_td = AtomNode.get_type_data(cfg.data.lig_graph)
        res_td = ProtResidueNode.get_type_data(cfg.data.rec_graph)
        cat_td = TensorTD((atom_td.cat_feat.shape[0] + res_td.cat_feat.shape[0],), dtype=torch.long, max_values=atom_td.cat_feat.max_values + res_td.cat_feat.max_values)
        scal_td = TensorTD((atom_td.scal_feat.shape[0] + res_td.scal_feat.shape[0],), dtype=torch.float32)
        coord_td = TensorTD((3,), dtype=torch.float32)
        return ClassTD(InteractionNode, coord=coord_td, cat_feat=cat_td, scal_feat=scal_td)

class InteractionEdge(Edge3d):

    @staticmethod
    def get_type_data(cfg: DictConfig):
        assert cfg.data.rec_graph.node_type == "residue"
        bond_td = BondEdge.get_type_data(cfg.data.lig_graph)
        prot_td = DistEdge.get_type_data(cfg.data.rec_graph)
        inter_td = DistEdge.get_type_data(cfg.data.interaction_graph)
        cat_td = TensorTD((bond_td.cat_feat.shape[0] + prot_td.cat_feat.shape[0] + inter_td.cat_feat.shape[0],), dtype=torch.long, max_values=bond_td.cat_feat.max_values + prot_td.cat_feat.max_values + inter_td.cat_feat.max_values)
        scal_td = TensorTD((bond_td.scal_feat.shape[0] + prot_td.scal_feat.shape[0] + inter_td.scal_feat.shape[0],), dtype=torch.float32)
        return ClassTD(InteractionEdge, cat_feat=cat_td, scal_feat=scal_td)

def merge_batches(batches, type):
    """ Helper function for merging graphs. Merges both items in
    edata and ndata"""
    ret = {}
    for attr in [ "cat_feat", "scal_feat" ]:
        default = torch.zeros((0,1))  if attr == "scal_feat" else torch.zeros((0,0))
        data = [ getattr(batch, attr) if isinstance(batch, Batch) else default for batch in batches]
        for d in data:
            assert len(d.shape) == 2
        n_items = sum([ d.shape[0] for d in data ])
        n_feat = sum([ d.shape[1] for d in data ])
        feat = torch.zeros((n_items, n_feat), device=data[0].device, dtype=data[0].dtype)
        
        ax1_idx = 0
        ax2_idx = 0
        for d in data:
            feat[ax1_idx:ax1_idx+d.shape[0], ax2_idx:ax2_idx+d.shape[1]] = d
            ax1_idx += d.shape[0]
            ax2_idx += d.shape[1]

        ret[attr] = feat
    if hasattr(batches[0], "coord"):
        ret["coord"] = torch.cat([ b.coord for b in batches], 0)
    return Batch(type, **ret)

def merge_graphs(inter_cfg, g1: MolGraph, g2: ProtGraph) -> Tuple[List[InteractionNode], Tuple[int, int], List[InteractionEdge]]:

    nodes = merge_batches([g1.ndata, g2.ndata], Node3d)

    edges = g1.edges
    num_g1_nodes = len(g1.ndata)
    for idx1, idx2 in g2.edges:
        edges.append((num_g1_nodes + idx1, num_g1_nodes + idx2))

    extra_edges = []
    extra_edata = []
    for (i,j) in make_knn_edgelist(nodes, inter_cfg.dist_cutoff, inter_cfg.max_neighbors):
        # only make edges between the og graphs
        if (i < num_g1_nodes) != (j < num_g1_nodes):
            extra_edges.append((i,j))
            node1 = nodes[i]
            node2 = nodes[j]
            extra_edata.append(DistEdge(inter_cfg, node1, node2))

    if len(extra_edata) > 0:
        extra_edata = make_batch(extra_edata)
    edges += extra_edges
    edata = merge_batches([g1.edata, g2.edata, extra_edata], Edge3d)

    assert edata.scal_feat.shape[-1] == 2
    assert edata.cat_feat.shape[-1] == 1

    return nodes, edges, edata

class InteractionGraph(Graph3d):
    
    @staticmethod
    def get_type_data(cfg: DictConfig) -> GraphTD:
        node_td = InteractionNode.get_type_data(cfg)
        edge_td = InteractionEdge.get_type_data(cfg)
        return GraphTD(InteractionGraph, node_td, edge_td, ShapeVar("LN"), ShapeVar("LE"))

    def __init__(self, cfg: DictConfig, lig_graph: MolGraph, rec_graph: ProtGraph):
        nodes, edges, edata = merge_graphs(cfg.data.interaction_graph, lig_graph, rec_graph)
        super().__init__(nodes, edges, edata)