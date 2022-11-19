
import torch
from typing import List, Tuple
from omegaconf import DictConfig
from rdkit import Chem
from Bio.PDB.Model import Model

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

def merge_graphs(inter_cfg, g1: MolGraph, g2: ProtGraph) -> Tuple[List[InteractionNode], Tuple[int, int], List[InteractionEdge]]:
    nodes = []
    for node in g1.ndata:
        more_cat_feat = torch.zeros_like(g2.ndata.cat_feat[0])
        more_scal_feat = torch.zeros_like(g2.ndata.scal_feat[0])
        nodes.append(InteractionNode(
            node.coord,
            torch.cat((node.cat_feat, more_cat_feat), 0),
            torch.cat((node.scal_feat, more_scal_feat), 0)
        ))

    for node in g2.ndata:
        more_cat_feat = torch.zeros_like(g1.ndata.cat_feat[0])
        more_scal_feat = torch.zeros_like(g1.ndata.scal_feat[0])
        nodes.append(InteractionNode(
            node.coord,
            torch.cat((more_cat_feat, node.cat_feat), 0),
            torch.cat((more_scal_feat, node.scal_feat), 0)
        ))

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

    edges += extra_edges

    edata = []
    for edge in g1.edata:
        more_cat_feat = torch.zeros_like(g2.edata.cat_feat[0])
        more_scal_feat = torch.zeros_like(g2.edata.scal_feat[0])
        extra_cat_feat = torch.zeros_like(extra_edata[0].cat_feat)
        extra_scal_feat = torch.zeros_like(extra_edata[0].scal_feat)
        edata.append(InteractionEdge(
            torch.cat((edge.cat_feat, more_cat_feat, extra_cat_feat), 0),
            torch.cat((edge.scal_feat, more_scal_feat, extra_scal_feat), 0)
        ))

    for edge in g2.edata:
        more_cat_feat = torch.zeros_like(g1.edata.cat_feat[0])
        more_scal_feat = torch.zeros_like(g1.edata.scal_feat[0])
        extra_cat_feat = torch.zeros_like(extra_edata[0].cat_feat)
        extra_scal_feat = torch.zeros_like(extra_edata[0].scal_feat)
        edata.append(InteractionEdge(
            torch.cat((more_cat_feat, edge.cat_feat, extra_cat_feat), 0),
            torch.cat((more_scal_feat, edge.scal_feat, extra_scal_feat), 0)
        ))

    for edge in extra_edata:
        more_cat_feat = torch.zeros_like(g1.edata.cat_feat[0])
        more_scal_feat = torch.zeros_like(g1.edata.scal_feat[0])
        extra_cat_feat = torch.zeros_like(g2.edata.cat_feat[0])
        extra_scal_feat = torch.zeros_like(g2.edata.scal_feat[0])
        edata.append(InteractionEdge(
            torch.cat((more_cat_feat, extra_cat_feat, edge.cat_feat), 0),
            torch.cat((more_scal_feat, extra_scal_feat, edge.scal_feat), 0)
        ))

    # print(len(extra_edata), len(g1.edata), len(g2.edata))

    return nodes, edges, edata

class InteractionGraph(Graph3d):
    
    @staticmethod
    def get_type_data(cfg: DictConfig) -> GraphTD:
        node_td = InteractionNode.get_type_data(cfg)
        edge_td = InteractionEdge.get_type_data(cfg)
        return GraphTD(InteractionGraph, node_td, edge_td, ShapeVar("LN"), ShapeVar("LE"))

    def __init__(self, cfg: DictConfig, lig: Chem.Mol, rec: Model):
        lig_graph = MolGraph(cfg, lig)
        rec_graph = ProtGraph(cfg, rec)
        nodes, edges, edata = merge_graphs(cfg.data.interaction_graph, lig_graph, rec_graph)
        super().__init__(nodes, edges, edata)