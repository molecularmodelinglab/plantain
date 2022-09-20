from omegaconf import DictConfig
from rdkit import Chem
import torch

from datasets.graphs.graph3d import Graph3d, Node3d, Edge3d
from datasets.utils import safe_index

from terrace.graph import GraphTD
from terrace.type_data import ClassTD, TensorTD, ShapeVar

ATOM_RADII_DICT = {
    'H': 1.1,
    'C': 1.7,
    'N': 1.55,
    'O': 1.52,
    'P': 1.8,
    'S': 1.8,
    'Cl': 1.75,
    'F': 1.47,
    'Br': 1.85,
    'I': 1.98
}

ATOM_COLOR_DICT = {
    'C': '#909090',
    'H': '#ffffff',
    'N': '#3050F8',
    'O': '#ff0D0D',
    'P': '#FF8000',
    'S': '#FFFF30',
    'Cl': '#1FF01F',
    'F': '#90E050',
    'Br': '#A62929',
    'I': '#940094'
}

def get_element(atom, mol):
    periodic_table = Chem.GetPeriodicTable()
    num = atom.GetAtomicNum()
    return Chem.PeriodicTable.GetElementSymbol(periodic_table, num)

def get_formal_charge(atom, mol):
    return atom.GetFormalCharge()
    
def get_hybridization(atom, mol):
    return str(atom.GetHybridization())

def get_is_aromatic(atom, mol):
    return atom.GetIsAromatic()

def get_numH(atom, mol):
    return atom.GetTotalNumHs()

possible_atom_feats = {
    "element": [ "H", "C", "N", "O", "F", "S", "P", "Cl", "Br", "I" ],
    "formal_charge": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    "hybridization": [ 'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'],
    "is_aromatic": [ True, False, 'misc' ],
    "numH": [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc']
}

def get_bond_order(bond, mol):
    return bond.GetBondType()

possible_bond_feats = {
    "bond_order": [ Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC ],
}

class AtomNode(Node3d):

    @staticmethod
    def get_type_data(mol_cfg: DictConfig) -> ClassTD:
        max_cat_vals = []
        for feat_name in mol_cfg.atom_feats:
            max_cat_vals.append(len(possible_atom_feats[feat_name]))
        cat_td = TensorTD((len(max_cat_vals), ), dtype=torch.long, max_values=max_cat_vals)
        scal_td = TensorTD((0,), dtype=torch.float32)
        coord_td = TensorTD((3,), dtype=torch.float32)
        return ClassTD(AtomNode, coord=coord_td, cat_feat=cat_td, scal_feat=scal_td)

    def __init__(self, mol_cfg: DictConfig, atom: Chem.Atom, mol: Chem.Mol):
        cat_feats = []
        scal_feats = []
        for feat_name in mol_cfg.atom_feats:
            get_feat = globals()["get_" + feat_name]
            possible = possible_atom_feats[feat_name]
            feat = safe_index(possible, get_feat(atom, mol))
            cat_feats.append(feat)
        
        coord = mol.GetConformer().GetPositions()[atom.GetIdx()]

        coord = torch.tensor(coord, dtype=torch.float32)
        cat_feats = torch.tensor(cat_feats, dtype=torch.long)
        scal_feats = torch.tensor(scal_feats, dtype=torch.float32)

        super(AtomNode, self).__init__(coord, cat_feats, scal_feats)

    def get_element(self) -> str:
        return ALLOWED_ELEMENTS[int(self.cat_feat[0])]

    def get_radius(self) -> float:
        # decrease radius so we can actually see the bonds
        return ATOM_RADII_DICT[self.get_element()] * 0.3

    def get_color(self) -> str:
        return ATOM_COLOR_DICT[self.get_element()]

class BondEdge(Edge3d):

    @staticmethod
    def get_type_data(mol_cfg: DictConfig) -> ClassTD:
        max_cat_vals = []
        for feat_name in mol_cfg.bond_feats:
            max_cat_vals.append(len(possible_bond_feats[feat_name]))
        cat_td = TensorTD((len(max_cat_vals), ), dtype=torch.long, max_values=max_cat_vals)
        scal_td = TensorTD((0,), dtype=torch.float32)
        return ClassTD(BondEdge, cat_feat=cat_td, scal_feat=scal_td)
    
    def __init__(self, mol_cfg: DictConfig, bond: Chem.Bond, mol: Chem.Mol):
        cat_feats = []
        scal_feats = []
        for feat_name in mol_cfg.bond_feats:
            get_feat = globals()["get_" + feat_name]
            possible = possible_bond_feats[feat_name]
            feat = safe_index(possible, get_feat(bond, mol))
            cat_feats.append(feat)
        
        cat_feats = torch.tensor(cat_feats, dtype=torch.long)
        scal_feats = torch.tensor(scal_feats, dtype=torch.float32)
        super(BondEdge, self).__init__(cat_feats, scal_feats)

class MolGraph(Graph3d):

    @staticmethod
    def get_type_data(cfg: DictConfig) -> GraphTD:
        mol_cfg = cfg.data.lig_graph
        atom_td = AtomNode.get_type_data(mol_cfg)
        bond_td = BondEdge.get_type_data(mol_cfg)
        return GraphTD(MolGraph, atom_td, bond_td, ShapeVar("LN"), ShapeVar("LE"))

    def __init__(self, cfg: DictConfig, mol: Chem.Mol):
        mol_cfg = cfg.data.lig_graph
        nodes = [AtomNode(mol_cfg, atom, mol) for atom in mol.GetAtoms()]

        edges = []
        edata = []
        for bond in mol.GetBonds():
            idx1 = bond.GetBeginAtomIdx()
            idx2 = bond.GetEndAtomIdx()
            edges.append((idx1, idx2))
            edata.append(BondEdge(mol_cfg, bond, mol))

        super(MolGraph, self).__init__(nodes, edges, edata)

def mol_graph_from_sdf(cfg, sdf_file):
    return MolGraph(cfg, next(Chem.SDMolSupplier(sdf_file, sanitize=True)))