from copy import deepcopy
import py3Dmol
from ipywidgets import interact, interactive, fixed
from rdkit import Chem
from common.utils import get_pdb_str
from common.pose_transform import add_pose_to_mol
from common.pose_transform import Pose

def add_rec_to_view(view, rec, style="surface"):
    view.addModel(get_pdb_str(rec))
    view.zoomTo()
    if style == "surface":
        view.addSurface(py3Dmol.VDW,{'opacity':0.7,'color':'white'}, {"chain": "A"})
    else:
        view.setStyle(style)
    return view

def add_lig_to_view(view, lig, pose):
    mol = deepcopy(lig)
    add_pose_to_mol(mol, pose)
    view.addModel(Chem.MolToMolBlock(mol))
    view.setStyle("stick")

# http://rdkit.blogspot.com/2016/07/using-ipywidgets-and-py3dmol-to-browse.html
# ^^^ very helpful reference
def show_multiple_lig_poses(rec, lig, poses):
    def drawit(pose_id):
        view = py3Dmol.view()
        add_lig_to_view(view, lig, Pose(coord=poses.coord[pose_id]))
        add_rec_to_view(view, rec)
        view.show()
    interact(drawit, pose_id=(0, len(poses.coord)-1))