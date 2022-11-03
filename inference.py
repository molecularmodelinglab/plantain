#!/usr/bin/env python

import argparse
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from terrace.comp_node import Input
from terrace.batch import make_batch_td, DataLoader

from datasets.inference_dataset import InferenceDataset
from datasets.data_types import IsActiveData
from models.make_model import make_model

def inference():

    parser = argparse.ArgumentParser(description="BANANA inference")
    parser.add_argument("smi_file", type=str, help="Input smiles (.smi) file of compounds to screen")
    parser.add_argument("pdb_file", type=str, help="Pocket pdb file")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers in the dataloader")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--no_gpu", dest='no_gpu', action='store_true', help="Disable GPU (only use CPU)")
    parser.add_argument("--out_file", type=str, default="out.txt", help="File to store the output scores")
    parser.set_defaults(no_gpu=False)
    args = parser.parse_args()

    device = "cpu" if args.no_gpu else "cuda:0"
    
    cfg = OmegaConf.load("configs/short_thicc_op_gnn.yaml")
    in_node = Input(make_batch_td(IsActiveData.get_type_data(cfg)))
    
    model = make_model(cfg, in_node)
    model.load_state_dict(torch.load("banana_final.pt"))
    model = model.to(device)
    model.eval()

    dataset = InferenceDataset(cfg, args.smi_file, args.pdb_file)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            num_workers=args.num_workers, pin_memory=True,
                            shuffle=False)

    with open(args.out_file, "w") as f:
        for batch in tqdm(dataloader):
            batch = batch.to(device)
            output = model(batch).cpu().numpy()
            for out in output:
                f.write(f"{out}\n")

if __name__ == "__main__":
    with torch.no_grad():
        inference()

