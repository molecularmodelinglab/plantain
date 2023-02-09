from typing import Optional
from omegaconf import DictConfig
from rdkit.Geometry import Point3D
from rdkit import Chem
import torch
from terrace import CategoricalTensor, collate

from .graph3d import Graph3d, Node3d, Edge3d
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

def get_element(atom, mol):
    periodic_table = Chem.GetPeriodicTable()
    num = atom.GetAtomicNum()
    return Chem.PeriodicTable.GetElementSymbol(periodic_table, num)

def get_formal_charge(atom, mol):
    return atom.GetFormalCharge()
    
def get_hybridization(atom, mol):
    return atom.GetHybridization()

def get_is_aromatic(atom, mol):
    return atom.GetIsAromatic()

def get_numH(atom, mol):
    return atom.GetTotalNumHs()

possible_atom_feats = {
    "element": [ None, "H", "C", "N", "O", "F", "S", "P", "Cl", "Br", "I", "misc" ],
    "formal_charge": [ None, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    "hybridization": [ None, Chem.HybridizationType.SP, Chem.HybridizationType.SP2, Chem.HybridizationType.SP3, Chem.HybridizationType.SP3D, Chem.HybridizationType.SP3D2, 'misc'],
    "is_aromatic": [ None, True, False, 'misc' ],
    "numH": [ None, 0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc']
}

def get_bond_order(bond, mol):
    return bond.GetBondType()

possible_bond_feats = {
    "bond_order": [ None, Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC ],
}

class AtomNode(Node3d):

    def __init__(self, atom_feats, atom: Chem.Atom, mol: Chem.Mol, conformer: Optional[int]):
        cat_feats = []
        scal_feats = []
        num_classes = []
        for feat_name in atom_feats:
            get_feat = globals()["get_" + feat_name]
            possible = possible_atom_feats[feat_name]
            feat = safe_index(possible, get_feat(atom, mol))
            num_classes.append(len(possible))
            cat_feats.append(feat)
        
        if conformer is not None:
            point = mol.GetConformer(conformer).GetAtomPosition(atom.GetIdx())
            coord = [ point.x, point.y, point.z ]
        else:
            coord = [0.0, 0.0, 0.0]

        coord = torch.tensor(coord, dtype=torch.float32)
        cat_feats = torch.tensor(cat_feats, dtype=torch.long)
        cat_feats = CategoricalTensor(cat_feats, num_classes=num_classes)
        scal_feats = torch.tensor(scal_feats, dtype=torch.float32)

        super(AtomNode, self).__init__(coord, cat_feats, scal_feats)

    def get_element(self) -> str:
        return possible_atom_feats["element"][int(self.cat_feat[0])]

    def get_radius(self) -> float:
        # decrease radius so we can actually see the bonds
        return ATOM_RADII_DICT[self.get_element()] * 0.3

    def get_color(self) -> str:
        return ATOM_COLOR_DICT[self.get_element()]

class BondEdge(Edge3d):
 
    def __init__(self, bond_feats, bond: Chem.Bond, mol: Chem.Mol):
        cat_feats = []
        scal_feats = []
        num_classes = []
        for feat_name in bond_feats:
            get_feat = globals()["get_" + feat_name]
            possible = possible_bond_feats[feat_name]
            num_classes.append(len(possible))
            feat = safe_index(possible, get_feat(bond, mol))
            cat_feats.append(feat)
        
        cat_feats = torch.tensor(cat_feats, dtype=torch.long)
        cat_feats = CategoricalTensor(cat_feats, num_classes=num_classes)
        scal_feats = torch.tensor(scal_feats, dtype=torch.float32)
        super(BondEdge, self).__init__(cat_feats, scal_feats)

class MolGraph(Graph3d):

    def __init__(self, cfg: DictConfig, mol: Chem.Mol, conformer: Optional[int] = 0):
        mol = Chem.RemoveHs(mol)
        mol_cfg = cfg.data.lig_graph
        atom_feats = list(mol_cfg.atom_feats)
        bond_feats = list(mol_cfg.bond_feats)
        nodes = [AtomNode(atom_feats, atom, mol, conformer) for atom in mol.GetAtoms()]

        edges = []
        edata = []
        for bond in mol.GetBonds():
            idx1 = bond.GetBeginAtomIdx()
            idx2 = bond.GetEndAtomIdx()
            edges.append((idx1, idx2))
            edata.append(BondEdge(bond_feats, bond, mol))

        super().__init__(nodes, edges, edata)

    def to_rdkit(self):
        """ Turns this bad boi into a nice rdkit mol"""
        mol = Chem.RWMol()

        for atom_feat in self.ndata.cat_feat:
            elem_idx = int(atom_feat[0])
            charge_idx = int(atom_feat[1])
            hybrid_idx = int(atom_feat[2])
            h_idx = int(atom_feat[4])
            elem = possible_atom_feats["element"][elem_idx]
            charge = possible_atom_feats["formal_charge"][charge_idx]
            hybrid = possible_atom_feats["hybridization"][hybrid_idx]
            numH = possible_atom_feats["numH"][h_idx]
            atom = Chem.Atom(elem)
            atom.SetFormalCharge(charge)
            atom.SetHybridization(hybrid)
            atom.SetNumExplicitHs(numH)
            mol.AddAtom(atom)

        seen = set()
        for (i, j), bond_idx in zip(self.edges, self.edata.cat_feat[:,0]):
            if (j, i) in seen: continue
            seen.add((i, j))
            order = possible_bond_feats["bond_order"][int(bond_idx)]
            mol.AddBond(i, j, order)

        conformer = Chem.Conformer(mol.GetNumAtoms())
        for i, coord in enumerate(self.ndata.coord):
            conformer.SetAtomPosition(i, Point3D(float(coord[0]),
                                                 float(coord[1]),
                                                 float(coord[2])))

        mol.AddConformer(conformer)
        Chem.SanitizeMol(mol)
        # Chem.AssignAtomChiralTagsFromStructure(mol)
        Chem.AssignStereochemistryFrom3D(mol)

        return mol

def get_mol_coords(mol, conf_id):
    conformer = mol.GetConformer(conf_id)
    ret = []
    for idx in range(mol.GetNumAtoms()):
        point = conformer.GetAtomPosition(idx)
        coord = [ point.x, point.y, point.z ]
        ret.append(coord)
    return torch.tensor(ret, dtype=torch.float32)

def mol_graph_from_sdf(cfg, sdf_file):
    return MolGraph(cfg, next(Chem.SDMolSupplier(sdf_file, sanitize=True)))