import warnings
from omegaconf import DictConfig
import torch
from Bio.PDB.Model import Model
from Bio.PDB.Residue import Residue
from Bio.PDB.Atom import Atom
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB import PDBParser
from terrace import CategoricalTensor

from terrace.batch import Batch, Batchable, NoStackTensor
from terrace.dataframe import DFRow
from terrace import NoStackCatTensor
from terrace.graph import Graph
from .graph3d import Graph3d, Node3d, Edge3d
from .knn import make_knn_edgelist, make_res_knn_edgelist
from .dist_edge import DistEdge
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
    "atom_type": [None, 'C', 'CA', 'CB', 'CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE2', 'CE3', 'CG', 'CG1', 'CG2', 'CH2',
                  'CZ', 'CZ2', 'CZ3', 'N', 'ND1', 'ND2', 'NE', 'NE1', 'NE2', 'NH1', 'NH2', 'NZ', 'O', 'OD1',
                  'OD2', 'OE1', 'OE2', 'OG', 'OG1', 'OH', 'OXT', 'SD', 'SG', 'misc'],
    "atom_residue_type": [None, 'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET',
                     'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'HIP', 'HIE', 'TPO', 'HID', 'LEV', 'MEU',
                     'PTR', 'GLV', 'CYT', 'SEP', 'HIZ', 'CYM', 'GLM', 'ASQ', 'TYS', 'CYX', 'GLZ', 'misc'],
}

possible_residue_feats = {
    "residue_type": [None, 'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET',
                     'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'HIP', 'HIE', 'TPO', 'HID', 'LEV', 'MEU',
                     'PTR', 'GLV', 'CYT', 'SEP', 'HIZ', 'CYM', 'GLM', 'ASQ', 'TYS', 'CYX', 'GLZ', 'misc'],
}

def get_atom_type(atom, rec):
    return atom.name

def get_residue_type(residue, rec):
    return residue.get_resname()

def get_atom_residue_type(atom, rec):
    return atom.get_parent().get_resname()

class ProtAtomNode(Node3d):

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

    def __init__(self, prot_cfg: DictConfig, residue: Residue, prot: Model):
        coord = None
        for atom in residue:
            if atom.name == "CA":
                coord = torch.tensor(list(atom.get_vector()), dtype=torch.float32)
        assert coord is not None
        cat_feat = []
        scal_feat = []
        num_classes = []
        for feat_name in prot_cfg.residue_feats:
            get_feat = globals()["get_" + feat_name]
            possible = possible_residue_feats[feat_name]
            num_classes.append(len(possible))
            feat = safe_index(possible, get_feat(residue, prot))
            cat_feat.append(feat)
        cat_feat = torch.tensor(cat_feat, dtype=torch.long)
        cat_feat = CategoricalTensor(cat_feat, num_classes=num_classes)
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
    # nodes = []
    # for chain in prot:
    #     for residue in chain:
    #         if residue.get_resname() not in possible_residue_feats["residue_type"]: continue
    #         if prot_cfg.node_type == "residue":
    #             nodes.append(ProtResidueNode(prot_cfg, residue, prot))
    #         elif prot_cfg.node_type == "atom":
    #             for atom in residue:
    #                 nodes.append(ProtAtomNode(prot_cfg, atom, prot))

    if prot_cfg.node_type == "residue":
        residues = []
        for chain in prot:
            for residue in chain:
                if residue.get_resname() not in possible_residue_feats["residue_type"]: continue
                residues.append(residue)

        
        cat_feat = torch.zeros((len(residues), len(prot_cfg.residue_feats)), dtype=torch.long)
        num_classes = [ len(possible_residue_feats[feat_name]) for feat_name in prot_cfg.residue_feats]
        coord = torch.zeros((len(residues), 3), dtype=torch.float32)

        for i, feat_name in enumerate(prot_cfg.residue_feats):
            get_feat = globals()["get_" + feat_name]
            possible = possible_residue_feats[feat_name]
            for j, residue in enumerate(residues):
                cat_feat[j, i] = safe_index(possible, get_feat(residue, prot))

        for j, residue in enumerate(residues):
            for atom in residue:
                if atom.name == "CA":
                    coord[j] = torch.tensor(list(atom.get_vector()), dtype=torch.float32)
                    break
            else:
                raise AssertionError("Failed to find alpha carbon in residue")

        cat_feat = CategoricalTensor(cat_feat, num_classes=num_classes)
        nodes = Batch(ProtResidueNode,
                    coord = coord,
                    cat_feat = cat_feat,
                    scal_feat = torch.zeros((len(residues), 0)))

    elif prot_cfg.node_type == "atom":
        atoms = []
        res_indexes = []
        for chain in prot:
            for i, residue in enumerate(chain):
                if residue.get_resname() not in possible_residue_feats["residue_type"]: continue
                for atom in residue:
                    atoms.append(atom)
                    res_indexes.append(i)

        cat_feat = torch.zeros((len(atoms), len(prot_cfg.atom_feats)), dtype=torch.long)
        num_classes = [ len(possible_atom_feats[feat_name]) for feat_name in prot_cfg.atom_feats]
        coord = torch.zeros((len(atoms), 3), dtype=torch.float32)

        for i, feat_name in enumerate(prot_cfg.atom_feats):
            get_feat = globals()["get_" + feat_name]
            possible = possible_atom_feats[feat_name]
            for j, atom in enumerate(atoms):
                cat_feat[j, i] = safe_index(possible, get_feat(atom, prot))


        for j, atom in enumerate(atoms):
            coord[j] = torch.tensor(list(atom.get_vector()), dtype=torch.float32)

        cat_feat = CategoricalTensor(cat_feat, num_classes=num_classes)
        nodes = Batch(ProtAtomNode,
                    coord = coord,
                    cat_feat = cat_feat,
                    scal_feat = torch.zeros((len(atoms), 0)))

    else:
        raise ValueError("rec cfg node_type must be either residue or atom")

    if prot_cfg.edge_method.type == "knn":
        edges, dists = make_knn_edgelist(nodes, 
                                         prot_cfg.edge_method.knn_rad,
                                         prot_cfg.edge_method.max_neighbors)
    elif prot_cfg.edge_method.type == "residue_knn":
        edges, dists = make_res_knn_edgelist(nodes,
                                             res_indexes,
                                             prot_cfg.edge_method.atom_knn_rad,
                                             prot_cfg.edge_method.residue_knn_rad)
    else:
        raise AssertionError()

    edata = DistEdge.make_from_dists(dists)

    return nodes, edges, edata

class ProtGraph(Graph3d):

    def __init__(self, cfg: DictConfig, prot: Model):
        prot_cfg = cfg.data.rec_graph

        nodes, edges, edata = get_nodes_and_edges_from_model(cfg, prot)

        # edata = []
        # for (i, j) in edges:
        #     node1 = nodes[i]
        #     node2 = nodes[j]
        #     edata.append(DistEdge(prot_cfg, node1, node2))

        super(ProtGraph, self).__init__(nodes, edges, edata, directed=True)

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
    nodes, edges, edata = get_nodes_and_edges_from_model(cfg, rec)
    return len(nodes), len(edges)


class FullRecData(Batchable):
    coord: torch.Tensor
    cat_feat: CategoricalTensor
    scal_feat: torch.Tensor
    res_index: torch.Tensor

    @staticmethod
    def collate_coord(x):
        return x

    @staticmethod
    def collate_cat_feat(x):
        return x

    @staticmethod
    def collate_scal_feat(x):
        return x

    @staticmethod
    def collate_res_index(x):
        return x

class FullRecNode(Node3d):
    _res_index: torch.Tensor

class FullRecGraph(Graph[FullRecNode, None]):
    
    # hacky way to get around that fact that we can't (atm) custom collate
    # ndata/edata objects in terrace
    def batch_get_res_index(self):
        ret = []
        cur_max_res = 0
        for node_slice in self.node_slices:
            cur_res = self.ndata._res_index[node_slice] 
            ret.append(cur_res + cur_max_res)
            cur_max_res += cur_res.amax() + 1
        return torch.cat(ret, 0)

def get_full_rec_data(cfg, rec):
    prot_cfg = cfg.data.rec_graph

    atoms = []
    res_indexes = []
    cur_res_index = 0
    for chain in rec:
        for i, residue in enumerate(chain):
            if residue.get_resname() not in possible_residue_feats["residue_type"]: continue
            for atom in residue:
                atoms.append(atom)
                res_indexes.append(cur_res_index)
            cur_res_index += 1

    cat_feat = torch.zeros((len(atoms), len(prot_cfg.atom_feats)), dtype=torch.long)
    num_classes = [ len(possible_atom_feats[feat_name]) for feat_name in prot_cfg.atom_feats]
    coord = torch.zeros((len(atoms), 3), dtype=torch.float32)

    for i, feat_name in enumerate(prot_cfg.atom_feats):
        get_feat = globals()["get_" + feat_name]
        possible = possible_atom_feats[feat_name]
        for j, atom in enumerate(atoms):
            cat_feat[j, i] = safe_index(possible, get_feat(atom, rec))


    for j, atom in enumerate(atoms):
        coord[j] = torch.tensor(list(atom.get_vector()), dtype=torch.float32)

    cat_feat = CategoricalTensor(cat_feat, num_classes=num_classes)
    scal_feat = torch.zeros((len(atoms), 0))
    res_indexes = torch.tensor(res_indexes, dtype=torch.long)

    if cfg.data.use_v2_full_rec_data:
        ndata = Batch(FullRecNode, 
                      coord=coord, 
                      cat_feat=cat_feat,
                      scal_feat=scal_feat,
                      _res_index=res_indexes)
        return FullRecGraph(ndata, [])
    else:
        return FullRecData(coord, 
                        cat_feat,
                        scal_feat,
                        res_indexes)