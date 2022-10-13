import warnings
from omegaconf import DictConfig
import torch
from Bio.PDB.Model import Model
from Bio.PDB.Residue import Residue
from Bio.PDB.Atom import Atom
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB import PDBParser

from terrace.type_data import ClassTD, TensorTD, ShapeVar
from terrace.graph import GraphTD
from datasets.graphs.graph3d import Graph3d, Node3d, Edge3d
from datasets.graphs.knn import make_knn_edgelist
from datasets.graphs.dist_edge import DistEdge
from datasets.utils import safe_index

# http://acces.ens-lyon.fr/biotic/rastop/help/colour.htm
RESIDUE_COLORS = {
    "ASP": "#E60A0A",
    "GLU": "#E60A0A",
    "CYS": "#E6E600",
    "MET": "#E6E600",
    "LYS": "#145AFF",
    "ARG": "#145AFF",
    "SER": "#FA9600",
    "THR": "#FA9600",
    "PHE": "#3232AA",
    "TYR": "#3232AA",
    "ASN":  "#00DCDC",
    "GLN": "#00DCDC",
    "GLY": "#EBEBEB",
    "LEU": "#0F820F",
    "VAL": "#0F820F",
    "ILE": "#0F820F",
    "ALA": "#C8C8C8",
    "TRP ":"#B45AB4",
    "HIS": "#8282D2",
    "PRO": "#DC9682",
    "misc": "#BEA06E",
}

possible_atom_feats = {
    "atom_type": ['C', 'CA', 'CB', 'CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE2', 'CE3', 'CG', 'CG1', 'CG2', 'CH2',
                  'CZ', 'CZ2', 'CZ3', 'N', 'ND1', 'ND2', 'NE', 'NE1', 'NE2', 'NH1', 'NH2', 'NZ', 'O', 'OD1',
                  'OD2', 'OE1', 'OE2', 'OG', 'OG1', 'OH', 'OXT', 'SD', 'SG', 'misc'],
}

possible_residue_feats = {
    "residue_type": ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET',
                     'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'HIP', 'HIE', 'TPO', 'HID', 'LEV', 'MEU',
                     'PTR', 'GLV', 'CYT', 'SEP', 'HIZ', 'CYM', 'GLM', 'ASQ', 'TYS', 'CYX', 'GLZ', 'misc'],
}

def get_atom_type(atom, rec):
    return atom.name

def get_residue_type(residue, rec):
    return residue.get_resname()

class ProtAtomNode(Node3d):

    @staticmethod
    def get_type_data(prot_cfg: DictConfig) -> ClassTD:
        max_cat_vals = []
        for feat_name in prot_cfg.atom_feats:
            max_cat_vals.append(len(possible_atom_feats[feat_name]))
        cat_td = TensorTD((len(max_cat_vals), ), dtype=torch.long, max_values=max_cat_vals)
        scal_td = TensorTD((0,), dtype=torch.float32)
        coord_td = TensorTD((3,), dtype=torch.float32)
        return ClassTD(ProtAtomNode, coord=coord_td, cat_feat=cat_td, scal_feat=scal_td)

    def __init__(self, prot_cfg: DictConfig, atom: Atom, prot: Model):
        coord = torch.tensor(list(atom.get_vector()), dtype=torch.float32)
        cat_feat = []
        scal_feat = []
        for feat_name in prot_cfg.atom_feats:
            get_feat = globals()["get_" + feat_name]
            possible = possible_atom_feats[feat_name]
            feat = safe_index(possible, get_feat(atom, prot))
            cat_feat.append(feat)
        cat_feat = torch.tensor(cat_feat, dtype=torch.long)
        scal_feat = torch.tensor(scal_feat, dtype=torch.float32)
        super(ProtAtomNode, self).__init__(coord, cat_feat, scal_feat)

class ProtResidueNode(Node3d):

    @staticmethod
    def get_type_data(prot_cfg: DictConfig) -> ClassTD:
        max_cat_vals = []
        for feat_name in prot_cfg.residue_feats:
            max_cat_vals.append(len(possible_residue_feats[feat_name]))
        cat_td = TensorTD((len(max_cat_vals), ), dtype=torch.long, max_values=max_cat_vals)
        scal_td = TensorTD((0,), dtype=torch.float32)
        coord_td = TensorTD((3,), dtype=torch.float32)
        return ClassTD(ProtResidueNode, coord=coord_td, cat_feat=cat_td, scal_feat=scal_td)

    def __init__(self, prot_cfg: DictConfig, residue: Residue, prot: Model):
        coord = None
        for atom in residue:
            if atom.name == "CA":
                coord = torch.tensor(list(atom.get_vector()), dtype=torch.float32)
        assert coord is not None
        cat_feat = []
        scal_feat = []
        for feat_name in prot_cfg.residue_feats:
            get_feat = globals()["get_" + feat_name]
            possible = possible_residue_feats[feat_name]
            feat = safe_index(possible, get_feat(residue, prot))
            cat_feat.append(feat)
        cat_feat = torch.tensor(cat_feat, dtype=torch.long)
        scal_feat = torch.tensor(scal_feat, dtype=torch.float32)
        super(ProtResidueNode, self).__init__(coord, cat_feat, scal_feat)

    def get_color(self):
        resname = possible_residue_feats["residue_type"][self.cat_feat[0]]
        if resname in RESIDUE_COLORS:
            return RESIDUE_COLORS[resname]
        else:
            return RESIDUE_COLORS["misc"]

def get_nodes_and_edges_from_model(cfg: DictConfig, prot: Model):
    prot_cfg = cfg.data.rec_graph
    nodes = []
    for chain in prot:
        for residue in chain:
            if residue.get_resname() not in possible_residue_feats["residue_type"]: continue
            if prot_cfg.node_type == "residue":
                nodes.append(ProtResidueNode(prot_cfg, residue, prot))
            elif prot_cfg.node_type == "atom":
                for atom in residue:
                    nodes.append(ProtAtomNode(prot_cfg, atom, prot))
    
    if prot_cfg.edge_method.type == "knn":
        edges = make_knn_edgelist(nodes, 
                                    prot_cfg.edge_method.knn_rad,
                                    prot_cfg.edge_method.max_neighbors)
    else:
        raise AssertionError()

    return nodes, edges

class ProtGraph(Graph3d):

    @staticmethod
    def get_type_data(cfg: DictConfig) -> GraphTD:
        prot_cfg = cfg.data.rec_graph
        if prot_cfg.node_type == "residue":
            node_td = ProtResidueNode.get_type_data(prot_cfg)
        elif prot_cfg.node_type == "atom":
            node_td = ProtAtomNode.get_type_data(prot_cfg)
        edge_td = DistEdge.get_type_data(prot_cfg)
        return GraphTD(ProtGraph, node_td, edge_td, ShapeVar("RN"), ShapeVar("RE"))

    def __init__(self, cfg: DictConfig, prot: Model):
        prot_cfg = cfg.data.rec_graph

        nodes, edges = get_nodes_and_edges_from_model(cfg, prot)

        edata = []
        for (i, j) in edges:
            node1 = nodes[i]
            node2 = nodes[j]
            edata.append(DistEdge(prot_cfg, node1, node2))

        super(ProtGraph, self).__init__(nodes, edges, edata)

def prot_graph_from_pdb(cfg, pdb_file):
    parser = PDBParser()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PDBConstructionWarning)
        structure = parser.get_structure('random_id', pdb_file)
        rec = structure[0]
    return ProtGraph(cfg, rec)

def get_node_and_edge_nums_from_pdb(cfg, pdb_file):
    parser = PDBParser()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PDBConstructionWarning)
        structure = parser.get_structure('random_id', pdb_file)
        rec = structure[0]
    nodes, edges = get_nodes_and_edges_from_model(cfg, rec)
    return len(nodes), len(edges)
