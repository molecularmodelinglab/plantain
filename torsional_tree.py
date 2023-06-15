import torch.functional as F
import torch
from openbabel import openbabel as ob
from openbabel import pybel
import networkx as nx
from copy import copy,deepcopy
import torchvision.transforms as transforms
import numpy as np
from scipy.spatial.transform import Rotation as R
from rdkit import Chem


class Node:
    def __init__(self,index, origin,atoms,parent=None):
        #atom indices are 1-indexed
        self.children = []
        self.parent = parent
        if self.parent is not None:
            self.parent-=1
        self.origin = origin -1
        self.origin_coords=None
        self.parent_coords=None
        self.atoms = atoms - 1
        self.axis = None
        self.coords = None
        self.index=index
        self.forces=None
        self.torques=None
        self.rotation=None
    
    def add_child(self, child):
        self.children.append(child)
    
    def set_derivative(self):
        
        if self.parent is not None:
            self.axis=self.parent_coords-self.origin_coords
            norm=check_norm(self.axis)
            self.axis=self.axis/norm
            self.rotation=torch.dot(self.torques,self.axis)
        else:
            self.rotation=self.torques

def check_norm(axis):
    norm=torch.norm(axis)
    if norm < 1e-6:
        norm = 1
    else:
        norm = norm
    return norm

def is_rot(bond,amide_bonds):
    #check if a bond is a rotatable bond
    #bond is an openbabel bond
    #return a boolean
    #if bond.GetBondOrder() != 1 or bond.IsInRing() or bond.IsAmide():
    if bond.GetBondOrder() != 1 or bond.IsInRing():
        return False
    src=bond.GetBeginAtomIdx() -1
    dst=bond.GetEndAtomIdx() - 1
    if [src,dst] in amide_bonds or [dst,src] in amide_bonds:
        return False
    else:
        #check if connected atoms have atleast 2 bonds
        src=pybel.Atom(bond.GetBeginAtom())
        dst=pybel.Atom(bond.GetEndAtom())
        if src.heavydegree < 2 or dst.heavydegree < 2:
            return False
    return True

def ob_copy_molecule(mol):
    mol_copy=ob.OBMol()
    mol_copy=pybel.Molecule(mol_copy)
    mol_copy.OBMol.BeginModify()
    for atom in ob.OBMolAtomIter(mol.OBMol):
        new_atom=mol_copy.OBMol.NewAtom()
        new_atom.SetAtomicNum(atom.GetAtomicNum())
        new_atom.SetVector(atom.GetVector())
        new_atom.SetIdx(atom.GetIdx())
    for bond in ob.OBMolBondIter(mol.OBMol):
        mol_copy.OBMol.AddBond(bond.GetBeginAtomIdx(),bond.GetEndAtomIdx(),bond.GetBondOrder())
    mol_copy.OBMol.EndModify()
    return mol_copy

def get_amides(mol):
        amide_bonds=[]
        smarts_expr = '[#7:1][C:2](=[O])'
        matches = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts_expr))
        for match in matches:
            amide_bonds.append([list(match)[0],list(match)[1]])
            amide_bonds.append([list(match)[1],list(match)[0]])
        return amide_bonds

def obtain_rigid_from_mol(mol,amide_bonds):
    #obtain a list of rigid fragments from a molecule
    #mol is an openbabel molecule
    #return a list of rigid fragments
    mol_pieces=ob_copy_molecule(mol)
    #iterate over bonds of molecule
    bonds_to_delete=[]
    for bond in ob.OBMolBondIter(mol.OBMol):
        if is_rot(bond,amide_bonds):
            bonds_to_delete.append(bond)
    if len(bonds_to_delete) > 0:
        for bond in bonds_to_delete:       
            bond_copy=mol_pieces.OBMol.GetBond(bond.GetBeginAtomIdx(),bond.GetEndAtomIdx())
            mol_pieces.OBMol.DeleteBond(bond_copy)
    #convert pybel molecule to network x graph
    rigid_fragments=get_rigid_fragments(mol_pieces)
    return rigid_fragments

def get_rigid_fragments(mol):
    mol_graph=to_networkx(mol)
    #obtain rigid fragments
    rigid_fragments=list(nx.connected_components(mol_graph))
    rigid_fragments=[list(frag) for frag in rigid_fragments]
    return rigid_fragments

def to_networkx(mol):
    mol_graph=nx.Graph()
    for atom in ob.OBMolAtomIter(mol.OBMol):
        mol_graph.add_node(atom.GetIdx())
    for bond in ob.OBMolBondIter(mol.OBMol):
        mol_graph.add_edge(bond.GetBeginAtomIdx(),bond.GetEndAtomIdx())
    return mol_graph

def get_root_fragment(mol,rigid_fragments):
    #get the root fragment of a molecule i.e. 
    #the fragment with the atom that gives the shortest maximal remaining subgraph

    #mol is an openbabel molecule
    #rigid_fragments is a list of rigid fragments
    #return the index of the rigid fragment
    best_root_atom=1
    smrs=mol.OBMol.NumAtoms()
    for atom in ob.OBMolAtomIter(mol.OBMol):
        smrs_temp=0
        frag_list=[]
        mol_copy = ob_copy_molecule(mol)
        atom_idx=atom.GetIdx()
        atom_copy=mol_copy.OBMol.GetAtom(atom_idx)
        mol_copy.OBMol.DeleteAtom(atom_copy)
        frag_list=get_rigid_fragments(mol_copy)
        for frag in frag_list:
            if len(frag) > smrs_temp:
                smrs_temp=len(frag)
        if smrs_temp < smrs:
            smrs=smrs_temp
            best_root_atom=atom_idx
    for i,frag in enumerate(rigid_fragments):
        if best_root_atom in frag:
            return i,best_root_atom

def breadth_first_generate(mol,rigid_fragments,root_index,best_root_atom):
    #perform a breadth first traversal of the rigid fragments
    #mol is an openbabel molecule
    #rigid_fragments is a list of rigid fragments
    #root_index is the index of the root fragment
    #return a list of the rigid fragments in the order of traversal
    visited=[]
    tree=[]
    queue=[]
    networkx_graph=to_networkx(mol)
    queue.append((root_index,best_root_atom,None))
    child_index=0
    while queue:
        root=queue.pop(0)
        node=Node(len(tree),root[1],np.array(rigid_fragments[root[0]]),root[2])
        tree.append(node)
        visited.append(root[0])
        for atom in rigid_fragments[root[0]]:
            for neighbor in networkx_graph.neighbors(atom):
                for i,frag in enumerate(rigid_fragments):
                    if neighbor in frag and i not in visited and i not in queue:
                        queue.append((i,neighbor,atom))
                        child_index+=1
                        node.add_child(child_index)
                        
    return tree,visited

def breadth_first_traversal(tree):
    #perform a breadth first traversal of a tree
    #tree is a list of nodes
    #return a list of the nodes in the order of traversal
    visited=[]
    queue=[]
    queue.append(tree[0])
    while queue:
        node=queue.pop(0)
        visited.append(node.index)
        for child in node.children:
            queue.append(tree[child])
    return visited

def set_coords(tree,coords):
    #set the coordinates of each node in a tree
    #tree is a list of nodes
    #coords is a dictionary of coordinates
    for node in tree:
        index=node.index
        node.coords=coords[node.atoms]
        node.origin_coords=coords[node.origin]
        if node.parent is not None:
            node.parent_coords=coords[node.parent]


def set_derivative(tree,tree_index,forces):
    #calculate forces(translation) and torques(rotation) for each node in a tree
    #effect of child's rotation, child's force and self rotation and forces taken into account
    node=tree[tree_index]
    node.forces=forces[node.atoms]
    node.torques=torch.cross(node.coords-node.origin_coords,node.forces)
    node.forces=torch.sum(node.forces,dim=0)
    node.torques=torch.sum(node.torques,dim=0)
    if len(node.children)==0:
        node.set_derivative()       
        return node.forces,node.torques
    else:
        for child in node.children:
            child_forces,child_torques = set_derivative(tree,child,forces)
            child_node = tree[child]
            node.forces = node.forces + child_forces
            r = child_node.origin_coords - node.origin_coords
            node.torques = node.torques + child_torques + torch.cross(r,child_forces)
        node.set_derivative()
        return node.forces,node.torques

def to_rotation_matrix(rotation_vector):
    # Extract the components of the rotation vector
    roll, pitch, yaw = rotation_vector[0], rotation_vector[1], rotation_vector[2]

    # Compute trigonometric values
    c_roll, s_roll = torch.cos(roll), torch.sin(roll)
    c_pitch, s_pitch = torch.cos(pitch), torch.sin(pitch)
    c_yaw, s_yaw = torch.cos(yaw), torch.sin(yaw)

    # Create the rotation matrix
    rotation_matrix = torch.zeros((3, 3), dtype=torch.float64)
    rotation_matrix[0, 0] = rotation_matrix[0,0]+ c_pitch * c_yaw
    rotation_matrix[0, 1] = rotation_matrix[0,1] + (-c_roll * s_yaw + s_roll * s_pitch * c_yaw)
    rotation_matrix[0, 2] = rotation_matrix[0,2] + (s_roll * s_yaw + c_roll * s_pitch * c_yaw)
    rotation_matrix[1, 0] = rotation_matrix[1,0] + (c_pitch * s_yaw)
    rotation_matrix[1, 1] = rotation_matrix[1,1] + (c_roll * c_yaw + s_roll * s_pitch * s_yaw)
    rotation_matrix[1, 2] = rotation_matrix[1,2] + (-s_roll * c_yaw + c_roll * s_pitch * s_yaw)
    rotation_matrix[2, 0] = rotation_matrix[2,0] + -s_pitch
    rotation_matrix[2, 1] = rotation_matrix[2,1] + (s_roll * c_pitch)
    rotation_matrix[2, 2] = rotation_matrix[2,2] + (c_roll * c_pitch)
    return rotation_matrix

def rotate_axis(axis,origin,coords,torsion):
    #rotate a set of coordinates about an axis
    #point1 is the first point of the axis
    #point2 is the second point of the axis
    #coords is a set of coordinates
    #torsion is the angle of rotation
    #return the rotated coordinates
    norm=check_norm(axis)
    axis=axis/norm
    rot_vec=axis*torsion
    rot_vec=rot_vec.detach().numpy()
    rotation_matrix=R.from_rotvec(rot_vec).as_matrix()
    rotation_matrix=torch.tensor(rotation_matrix)
    coords=coords-origin
    coords=coords@rotation_matrix
    coords=coords+origin
    return coords

'''def get_component(axis1,axis2,torsion,coords):
    #get the component of the torsion about a particular axis
    #axis1 is the first axis
    #axis2 is the second axis
    #torsion is the angle of rotation
    #coords is a set of coordinates
    #return the component of the torsion about the axis
    axis1=axis1/torch.linalg.norm(axis1)
    axis2=axis2/torch.linalg.norm(axis2)
    rot_vec=axis1*torsion
    rotation_matrix=to_rotation_matrix(rot_vec)
    coords=coords@rotation_matrix
    return coords@axis2

def set_conf(tree,torsions,tree_index,torsion_index,coords):
    #mostly convoluted because we want gradients from the parent node and the child node

    node=tree[tree_index]
    if len(node.children)==0:
        torsion=np.random.uniform(low=-np.pi,high=np.pi)
        torsion=torch.tensor(torsion,requires_grad=True)
        axis=coords[node.parent] - coords[node.origin]
        node.coords=rotate_axis(axis,node.coords,torsion)
        return torsions[torsion_index]-torsion, axis
    else:
        prev_torsion=0
        original_origin=coords[node.origin].clone().detach()
        for child in node.children:
            child_torsion,child_axis=set_conf(tree,torsions,child,torsion_index,coords)
            node.coords=rotate_axis(child_axis,node.coords,child_torsion)
            #get the rigid fragment back to its orginal origin 
            node.coords=node.coords+original_origin-coords[node.origin]
'''

#def calculate_derivative(tree,torsions.tree_index,torsion_index,coords):

def calculate_torsion_angle(coord1, coord2, coord3, coord4):

    # Calculate the vectors
    v1 = coord1 - coord2
    v2 = coord2 - coord3
    v3 = coord3 - coord4

    # Calculate the normal vectors
    n1 = torch.cross(v1, v2)
    n2 = torch.cross(v2, v3)

    # Normalize the normal vectors
    n1 = n1 / torch.norm(n1)
    n2 = n2 / torch.norm(n2)

    # Calculate the dot product of the normalized normal vectors
    dot_product = torch.dot(n1, n2)

    # Calculate the torsion angle in radians
    torsion_angle = torch.atan2(torch.norm(torch.cross(n1, n2)), dot_product)

    return torsion_angle


#Code thanks to Matt Ragoza https://github.com/mattragoza/LiGAN/blob/master/ligan/molecules.py

def rd_mol_to_ob_mol(rd_mol):
    '''
    Convert an RWMol to an OBMol, copying
    over the elements, coordinates, formal
    charges, bonds and aromaticity.
    '''
    ob_mol = ob.OBMol()
    ob_mol.BeginModify()
    rd_conf = rd_mol.GetConformer()

    for idx, rd_atom in enumerate(rd_mol.GetAtoms()):

        ob_atom = ob_mol.NewAtom()
        ob_atom.SetAtomicNum(rd_atom.GetAtomicNum())
        ob_atom.SetFormalCharge(rd_atom.GetFormalCharge())
        ob_atom.SetAromatic(rd_atom.GetIsAromatic())
        ob_atom.SetImplicitHCount(rd_atom.GetNumExplicitHs())

        rd_coords = rd_conf.GetAtomPosition(idx)
        ob_atom.SetVector(rd_coords.x, rd_coords.y, rd_coords.z)

    for rd_bond in rd_mol.GetBonds():

        # OB uses 1-indexing, rdkit uses 0
        i = rd_bond.GetBeginAtomIdx() + 1
        j = rd_bond.GetEndAtomIdx() + 1

        bond_type = rd_bond.GetBondType()
        if bond_type == Chem.BondType.SINGLE:
            bond_order = 1
        elif bond_type == Chem.BondType.DOUBLE:
            bond_order = 2
        elif bond_type == Chem.BondType.TRIPLE:
            bond_order = 3
        #openbabel does not have an aromatic bond order but who cares about valencies
        elif bond_type == Chem.BondType.AROMATIC:
            bond_order=5
        #amide bonds don't get detected by openbabel easily
        #if [i-1,j-1] in amide_bonds or [j-1,i-1] in amide_bonds:
        #    bond_order=2
        ob_mol.AddBond(i, j, bond_order)
        ob_bond = ob_mol.GetBond(i, j)
        ob_bond.SetAromatic(rd_bond.GetIsAromatic())

    ob_mol.EndModify()
    return ob_mol

def get_tree(rdmol,amide_bonds):
    ob_mol=rd_mol_to_ob_mol(rdmol)
    mol=pybel.Molecule(ob_mol)
    mol.removeh()
    rigid_fragments=obtain_rigid_from_mol(mol,amide_bonds)
    root_fragment,root_atom=get_root_fragment(mol,rigid_fragments)
    tree,visited=breadth_first_generate(mol,rigid_fragments,root_fragment,root_atom)
    return tree

def get_tree_edge_mapping(tree,rotate_edges):
    mapping=[]
    for j in rotate_edges:
        for i in range(1,len(tree)):
            node=tree[i]
            if node.parent in j:
                if node.origin in j:
                    mapping.append((i))
    return mapping

if __name__ == '__main__':
    test_smile="C#CCOCCOCCOCCNc1nc(N2CCN(C(=O)[C@H](CCC(=O)O)n3cc(C(N)CO)nn3)CC2)nc(N2CCN(C(=O)[C@H](CCC(=O)O)n3cc(C([NH3+])CO)nn3)CC2)n1"
    mol=pybel.readstring("smi",test_smile)
    mol.removeh()
    rigid_fragments=obtain_rigid_from_mol(mol)
    root_fragment,root_atom=get_root_fragment(mol,rigid_fragments)
    tree,visited=breadth_first_generate(mol,rigid_fragments,root_fragment,root_atom)
    mol.make3D()
    mol.removeh()
    coords=[]
    for atom in ob.OBMolAtomIter(mol.OBMol):
        x = atom.GetX()
        y = atom.GetY()
        z = atom.GetZ()
        coords.append([x,y,z])
    coords=np.array(coords)
    coords_tensor=torch.tensor(coords)
    #print(coords_tensor.shape)
    num_torsions=len(visited)-1
    #print(num_torsions)
    torsions=np.random.uniform(low=-np.pi,high=np.pi,size=(num_torsions))
    #print(torsions)
    set_coords(tree,coords_tensor)
    #for node in tree:
    #    print(node.coords,node.origin_coords,node.index,node.atoms,node.origin,node.parent,node.children,node.parent_coords)
    forces=np.random.uniform(low=0,high=5,size=(len(coords),3))
    forces=torch.tensor(forces,dtype=torch.float64,requires_grad=True)
    set_derivative(tree,0,forces)
    #for node in tree:
    #    print(node.rotation)
    #    forces_grad=torch.autograd.grad(node.rotation.sum(),forces)
    #    print(forces_grad)
    #    break
    new_mol_smile='CCCC'
    new_mol=pybel.readstring("smi",new_mol_smile)
    new_mol.make3D()
    new_mol.removeh()
    new_coords=[]
    for atom in ob.OBMolAtomIter(new_mol.OBMol):
        x = atom.GetX()
        y = atom.GetY()
        z = atom.GetZ()
        new_coords.append([x,y,z])
    new_coords=np.array(new_coords)
    new_coords_tensor=torch.tensor(new_coords,requires_grad=True)
    rigid_fragments=obtain_rigid_from_mol(new_mol)
    root_fragment,root_atom=get_root_fragment(new_mol,rigid_fragments)
    new_tree,new_visited=breadth_first_generate(new_mol,rigid_fragments,root_fragment,root_atom)
    print(calculate_torsion_angle(new_coords_tensor[0],new_coords_tensor[1],new_coords_tensor[2],new_coords_tensor[3]))
    axis=new_coords_tensor[2]-new_coords_tensor[1]
    with torch.no_grad():
        print(new_coords_tensor)
        new_coords_tensor[0:2]=rotate_axis(axis,new_coords_tensor[1],new_coords_tensor[0:2],np.pi/2)
        print(new_coords_tensor)
    print(calculate_torsion_angle(new_coords_tensor[0],new_coords_tensor[1],new_coords_tensor[2],new_coords_tensor[3]))
    set_coords(new_tree,new_coords_tensor)
    current_torsion=calculate_torsion_angle(new_coords_tensor[0],new_coords_tensor[1],new_coords_tensor[2],new_coords_tensor[3])
    desired_torsion=torch.pi
    diff=(current_torsion-desired_torsion)**2
    coord_grad=torch.autograd.grad(diff,new_coords_tensor)
    print(coord_grad[0])
    set_derivative(new_tree,0,coord_grad[0])
    print(new_tree[1].rotation)

    rotation_vector =np.array([0.0, 0.0, np.pi/2])
    rotation_matrix = R.from_rotvec(rotation_vector).as_matrix()
    rotation_matrix = torch.tensor(rotation_matrix)
    with torch.no_grad():
        #new_coords_tensor_rot=new_coords_tensor - new_coords_tensor[1]
        new_coords_tensor_rot=new_coords_tensor@rotation_matrix
        #new_coords_tensor_rot=new_coords_tensor_rot + new_coords_tensor[1]
    coords_inverse=torch.pinverse(new_coords_tensor)
    random_rotation=coords_inverse@new_coords_tensor_rot
    print(random_rotation,rotation_matrix)
    coord_grad=torch.autograd.grad(random_rotation.sum(),new_coords_tensor)
    set_derivative(new_tree,0,coord_grad[0])
    print(new_tree[0].rotation)

