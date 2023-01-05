from rdkit import Chem
from rdkit.Chem import AllChem
import torch
import numpy as np
from traceback import print_exc

from terrace.type_data import TensorTD
from datasets.bigbind import BigBindDataset

class BigBindFpDataset(BigBindDataset):

    """ Returns the mol fingerprint and the activity, no receptor at all
    (Useful for the fp regressor baseline) """

    def __init__(self, cfg, split):
        super(BigBindFpDataset, self).__init__(cfg, "bigbind_fp", split)

    def __len__(self):
        return len(self.activities)

    def get_lig_file(self, index):
        return self.dir + "/" + self.activities.lig_file[index]

    def get_cache_key(self, index):

        lig_file = self.get_lig_file(index).split("/")[-1]
        activity = self.activities.pchembl_value[index]

        return lig_file + "_" + str(activity)

    def get_item_pre_cache(self, index):
        
        smiles = self.activities.lig_smiles[index]
        activity = torch.tensor(self.activities.pchembl_value[index], dtype=torch.float32)
        lig = Chem.MolFromSmiles(smiles)

        if self.cfg.data.fp_type == "morgan":
            lig_fp = np.zeros((0,), dtype=np.float32)
            Chem.DataStructs.ConvertToNumpyArray(
                AllChem.GetMorganFingerprintAsBitVect(
                    lig,
                    self.cfg.data.morgan_radius,
                    self.cfg.data.morgan_bits),
                lig_fp)
            lig_fp = torch.tensor(lig_fp)

        return lig_fp, activity

    def get_variance(self):
        return {
            "activity": self.activities.pchembl_value.var(),
        }

    def get_type_data(self):
        return (TensorTD((self.cfg.data.morgan_bits,)), TensorTD(tuple()))