from omegaconf import DictConfig
from rdkit import Chem
import torch

from datasets.graphs.graph3d import Graph3d, Node3d, Edge3d
from datasets.utils import safe_index

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

BOND_TYPE_LIST = [ Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC ]
BOND_TYPE_MAP = { bond: i for i, bond in enumerate(BOND_TYPE_LIST) }

ALLOWED_ELEMENTS = [ "H", "C", "N", "O", "F", "S", "P", "Cl", "Br", "I" ]

POSSIBLE_HYBRIDIZATIONS = [ 'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc']
POSSIBLE_FORMAL_CHARGES = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc']
POSSIBLE_NUM_H = [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc']

class AtomNode(Node3d):

    def __init__(self, mol_cfg: DictConfig, atom: Chem.Atom, mol: Chem.Mol):
        cat_feats = []
        scal_feats = []
        for feat_name in mol_cfg.atom_feats:
            if feat_name == "element":
                periodic_table = Chem.GetPeriodicTable()
                num = atom.GetAtomicNum()
                sym = Chem.PeriodicTable.GetElementSymbol(periodic_table, num)
                # don't do safe_index because we know the elements in our dataset
                # if I was wrong I want to know
                idx = ALLOWED_ELEMENTS.index(sym)
                cat_feats.append(idx)
            elif feat_name == "formal_charge":
                cat_feats.append(safe_index(POSSIBLE_FORMAL_CHARGES, atom.GetFormalCharge()))
            elif feat_name == "hybridization":
                hybrid = str(atom.GetHybridization())
                cat_feats.append(safe_index(POSSIBLE_HYBRIDIZATIONS, hybrid))
            elif feat_name == "is_aromatic":
                cat_feats.append(int(atom.GetIsAromatic()))
            elif feat_name == "numH":
                cat_feats.append(safe_index(POSSIBLE_NUM_H, atom.GetTotalNumHs()))
            else:
                raise AssertionError()
        
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
    
    def __init__(self, mol_cfg: DictConfig, bond: Chem.Bond, mol: Chem.Mol):
        cat_feats = []
        scal_feats = []
        for feat_name in mol_cfg.bond_feats:
            if feat_name == "bond_order":
                cat_feats.append(BOND_TYPE_MAP[bond.GetBondType()])
        
        cat_feats = torch.tensor(cat_feats, dtype=torch.long)
        scal_feats = torch.tensor(scal_feats, dtype=torch.float32)
        super(BondEdge, self).__init__(cat_feats, scal_feats)

class MolGraph(Graph3d):

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