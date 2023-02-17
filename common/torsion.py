from copy import deepcopy
import torch
import roma
import math
import networkx as nx
from rdkit import Chem
from terrace import Batchable

# many of these functions are takem/modified from:
#  https://github.com/HannesStark/EquiBind/blob/main/commons/geometry_utils.py
# and https://github.com/gcorso/torsional-diffusion/blob/master/utils/torsion.py

def rigid_transform_Kabsch_3D_torch(A, B):
    assert A.shape[1] == B.shape[1]
    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")
    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")


    # find mean column wise: 3 x 1
    centroid_A = torch.mean(A, axis=1, keepdims=True)
    centroid_B = torch.mean(B, axis=1, keepdims=True)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ Bm.T

    # find rotation
    U, S, Vt = torch.linalg.svd(H)

    R = Vt.T @ U.T

    # special reflection case
    if torch.linalg.det(R) < 0:
        # print("det(R) < R, reflection detected!, correcting for it ...")
        SS = torch.diag(torch.tensor([1.,1.,-1.], device=A.device))
        R = (Vt.T @ SS) @ U.T
    assert math.fabs(torch.linalg.det(R) - 1) < 1e-5

    t = -R @ centroid_A + centroid_B
    return R, t

def rigid_align(A, B):
    R, t = rigid_transform_Kabsch_3D_torch(A.T, B.T)
    return (R@A.T + t).T

def mol_to_nx(mol):
    G = nx.Graph()
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(),
                   bond.GetEndAtomIdx(),
                   bond_type=bond.GetBondType())
    return G

class TorsionData(Batchable):
    rot_edges: torch.Tensor
    rot_masks: torch.Tensor

    @staticmethod
    def collate_rot_edges(x):
        return x

    @staticmethod
    def collate_rot_masks(x):
        return x

    def from_mol(mol):
        G = mol_to_nx(mol)
        rot_edges = []
        to_rotate = []
        for edge in G.edges:
            bond_type = G.get_edge_data(*edge)["bond_type"]
            G2 = G.to_undirected()
            G2.remove_edge(*edge)
            if not nx.is_connected(G2) and bond_type == Chem.BondType.SINGLE:
                l = list(sorted(nx.connected_components(G2), key=len)[0])
                if len(l) > 1:
                    to_rotate.append(l)
                    rot_edges.append(edge)
            to_rotate.append([])

        mask_edges = torch.asarray([0 if len(l) == 0 else 1 for l in to_rotate], dtype=bool)
        mask_rotate = torch.zeros((torch.sum(mask_edges), len(G.nodes())), dtype=bool)
        idx = 0
        for i in range(len(G.edges())):
            if mask_edges[i]:
                mask_rotate[idx][torch.asarray(to_rotate[i], dtype=int)] = True
                idx += 1

        rot_edges = torch.asarray(rot_edges, dtype=torch.long)
        return TorsionData(rot_edges, mask_rotate)

    def set_angle(self, rot_index, angle, coord):
        idx1, idx2 = self.rot_edges[rot_index]
        mask = self.rot_masks[rot_index]
        if (isinstance(angle, float) or angle.shape == tuple()) and len(coord.shape) == 2:
            ax = coord[idx2] - coord[idx1]
            ax_norm = ax/torch.linalg.norm(ax)
            rot = roma.rotvec_to_rotmat(ax_norm*angle)
            rotated = torch.einsum('ij,bj->bi', rot, (to_rotate - coord[idx2])) + coord[idx2]
            ret = coord.clone()
            ret[mask] = rotated
            return ret
        elif len(angle.shape) == 1 and len(coord.shape) == 3:
            ax = coord[:,idx2] - coord[:,idx1]
            ax_norm = ax/torch.linalg.norm(ax, axis=1, keepdim=True)
            rotvec = torch.einsum('t,td->td', angle, ax_norm)
            rot = roma.rotvec_to_rotmat(rotvec)
            to_rotate = coord[:,mask]
            center = coord[:,idx2].unsqueeze(1)
            rotated = torch.einsum('tij,tbj->tbi', rot, (to_rotate - center)) + center
            ret = coord.clone()
            ret[:,mask] = rotated
            return ret
        elif len(angle.shape) == 1 and len(coord.shape) == 2:
            ax = coord[idx2] - coord[idx1]
            ax_norm = ax/torch.linalg.norm(ax)
            rotvec = torch.einsum('t,d->td', angle, ax_norm)
            rot = roma.rotvec_to_rotmat(rotvec)
            to_rotate = coord[mask]
            center = coord[idx2]
            rotated = torch.einsum('tij,bj->tbi', rot, (to_rotate - center)) + center
            ret = coord.expand(len(angle), -1, -1).clone()
            ret[:, mask] = rotated
            return ret
        else:
            raise NotImplementedError

    def set_all_angles(self, angles, coord):
        new_coord = coord
        for idx in range(angles.shape[-1]):
            new_coord = self.set_angle(idx, angles[...,idx], new_coord)
        return new_coord # rigid_align(new_coord, coord)