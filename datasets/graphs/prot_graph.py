from omegaconf import DictConfig
import torch
from Bio.PDB.Model import Model
from Bio.PDB.Residue import Residue
from Bio.PDB.Atom import Atom

from datasets.graphs.graph3d import Graph3d, Node3d, Edge3d
from datasets.graphs.knn import make_knn_edgelist
from datasets.graphs.dist_edge import DistEdge
from datasets.utils import safe_index

PROT_ATOM_TYPES = ['C', 'CA', 'CB', 'CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE2', 'CE3', 'CG', 'CG1', 'CG2', 'CH2',
                   'CZ', 'CZ2', 'CZ3', 'N', 'ND1', 'ND2', 'NE', 'NE1', 'NE2', 'NH1', 'NH2', 'NZ', 'O', 'OD1',
                   'OD2', 'OE1', 'OE2', 'OG', 'OG1', 'OH', 'OXT', 'SD', 'SG', 'misc']

PROT_RESIDUE_TYPES = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET',
                      'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'HIP', 'HIE', 'TPO', 'HID', 'LEV', 'MEU',
                      'PTR', 'GLV', 'CYT', 'SEP', 'HIZ', 'CYM', 'GLM', 'ASQ', 'TYS', 'CYX', 'GLZ', 'misc']

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

class ProtAtomNode(Node3d):

    def __init__(self, prot_cfg: DictConfig, atom: Atom, prot: Model):
        coord = torch.tensor(list(atom.get_vector()), dtype=torch.float32)
        cat_feat = []
        scal_feat = []
        for feat_name in prot_cfg.atom_feats:
            if feat_name == "type":
                cat_feat.append(safe_index(PROT_ATOM_TYPES, atom.name))
            else:
                print(feat_name, feat_name == "type")
                raise AssertionError()
        cat_feat = torch.tensor(cat_feat, dtype=torch.long)
        scal_feat = torch.tensor(scal_feat, dtype=torch.float32)
        super(ProtAtomNode, self).__init__(coord, cat_feat, scal_feat)

class ProtResidueNode(Node3d):

    def __init__(self, prot_cfg: DictConfig, residue: Residue, prot: Model):
        coord = None
        for atom in residue:
            if atom.name == "CA":
                coord = torch.tensor(list(atom.get_vector()), dtype=torch.float32)
        assert coord is not None
        cat_feat = []
        scal_feat = []
        for feat_name in prot_cfg.residue_feats:
            if feat_name == "type":
                cat_feat.append(safe_index(PROT_RESIDUE_TYPES, residue.get_resname()))
            else:
                raise AssertionError()
        cat_feat = torch.tensor(cat_feat, dtype=torch.long)
        scal_feat = torch.tensor(scal_feat, dtype=torch.float32)
        super(ProtResidueNode, self).__init__(coord, cat_feat, scal_feat)

    def get_color(self):
        resname = PROT_RESIDUE_TYPES[self.cat_feat[0]]
        if resname in RESIDUE_COLORS:
            return RESIDUE_COLORS[resname]
        else:
            return RESIDUE_COLORS["misc"]

class ProtGraph(Graph3d):

    def __init__(self, cfg: DictConfig, prot: Model):
        prot_cfg = cfg.data.rec_graph

        nodes = []
        for chain in prot:
            for residue in chain:
                if residue.get_resname() not in PROT_RESIDUE_TYPES: continue
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

        edata = []
        for (i, j) in edges:
            node1 = nodes[i]
            node2 = nodes[j]
            edata.append(DistEdge(prot_cfg, node1, node2))

        super(ProtGraph, self).__init__(nodes, edges, edata)