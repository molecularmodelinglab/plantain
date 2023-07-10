from rdkit import Chem
from rdkit.Chem import AllChem
from datasets.base_datasets import Dataset
from common.pose_transform import Pose
from common.utils import get_prot_from_file_no_cache
from data_formats.graphs.mol_graph import get_mol_coords
from terrace.dataframe import DFRow


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

        mol = Chem.MolFromSmiles(self.smiles[index])
        assert mol is not None
        mol = Chem.AddHs(mol)
        conf_id = AllChem.EmbedMolecule(mol)
        assert conf_id == 0
        success = AllChem.UFFOptimizeMolecule(mol)
        assert success

        mol = Chem.RemoveHs(mol)

        fake_pose = Pose(get_mol_coords(mol, 0))

        x = DFRow(lig=mol,
                  rec=self.rec)

        y = DFRow(lig_crystal_pose=fake_pose)

        return x, y