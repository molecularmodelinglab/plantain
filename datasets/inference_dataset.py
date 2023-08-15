from rdkit import Chem
from rdkit.Chem import AllChem
from datasets.base_datasets import Dataset
from common.pose_transform import Pose
from common.utils import get_prot_from_file_no_cache
from data_formats.graphs.mol_graph import get_mol_coords
from terrace.dataframe import DFRow

def get_and_embed_mol(smiles):
    """ Returns None if anything goes wrong """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    mol = Chem.AddHs(mol)
    conf_id = AllChem.EmbedMolecule(mol)
    if conf_id != 0: return None
    niter = 500
    success = AllChem.UFFOptimizeMolecule(mol, niter)
    if success != 0:
        print(f"Warning: UFF optimization failed to converge in {niter} iterations. This probably won't affect PLANTAIN's performance too much.")

    mol = Chem.RemoveHs(mol)
    return mol

class InferenceDataset(Dataset):
    """ Dataset for inference script. Just loads a smi file and a pdb file,
    and runs with it """

    def __init__(self, cfg, smi_file, pdb_file, transform):
        super().__init__(cfg, transform)
        self.rec = get_prot_from_file_no_cache(pdb_file)
        with open(smi_file, "r") as f:
            self.smiles = [ line.strip() for line in f ]

    def __len__(self):
        return len(self.smiles)

    def getitem_impl(self, index):

        success = True
        mol = get_and_embed_mol(self.smiles[index])
        if mol is None:
            mol = get_and_embed_mol("c1ccccc1")
            assert mol is not None
            success = False

        fake_pose = Pose(get_mol_coords(mol, 0))

        x = DFRow(lig=mol,
                  rec=self.rec)

        y = DFRow(success=success, lig_crystal_pose=fake_pose)

        return x, y

class DummyDataset(Dataset):
    """ Dummy dataset to initialize pretrained model """

    def __init__(self, cfg, transform):
        super().__init__(cfg, transform)
        self.rec = get_prot_from_file_no_cache("data/dummy_rec.pdb")

    def __len__(self):
        return 1

    def getitem_impl(self, index):
        mol = get_and_embed_mol("c1ccccc1")
        assert mol is not None

        fake_pose = Pose(get_mol_coords(mol, 0))

        x = DFRow(lig=mol,
                  rec=self.rec)

        y = DFRow(lig_crystal_pose=fake_pose)

        return x, y