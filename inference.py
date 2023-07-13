import argparse
import os
from tqdm import tqdm
from rdkit import Chem
from common.cfg_utils import get_config
from common.pose_transform import add_multi_pose_to_mol
from datasets.inference_dataset import InferenceDataset
from models.pretrained_plantain import get_pretrained_plantain
from terrace import collate

def inference():

    parser = argparse.ArgumentParser(description="PLANTAIN inference")
    parser.add_argument("smi_file", type=str, help="Input smiles (.smi) file of compounds to screen")
    parser.add_argument("pdb_file", type=str, help="Pocket pdb file")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers in the dataloader")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--no_gpu", dest='no_gpu', action='store_true', help="Disable GPU (only use CPU)")
    parser.add_argument("--out", type=str, default="predictions", help="Output folder for pose prediction")
    parser.set_defaults(no_gpu=False)
    args = parser.parse_args()

    device = "cpu" if args.no_gpu else "cuda:0"

    cfg = get_config("icml")
    model = get_pretrained_plantain()
    dataset = InferenceDataset(cfg, args.smi_file, args.pdb_file, model.get_input_feats())

    model = model.to(device)
    model.eval()

    os.makedirs(args.out, exist_ok=True)

    for i, (x, y) in enumerate(tqdm(dataset)):
        if not y.success:
            print(f"Something's wrong with compound at index {i}, skipping...")
            continue

        batch = collate([x]).to(device)
        pred = model(batch)[0]

        mol = x.lig
        add_multi_pose_to_mol(mol, pred.lig_pose)
        pose_file = f"{args.out}/{i}.sdf"
        writer = Chem.SDWriter(pose_file)
        for c in range(mol.GetNumConformers()):
            writer.write(mol, c)
        writer.close()

if __name__ == "__main__":
    inference()

    