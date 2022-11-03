
import torch

from tqdm import tqdm
from time import time
import wandb

from terrace.batch import DataLoader
from models.val_model import OldModel
from common.cfg_utils import get_config, get_run_config
from datasets.pdbbind import PDBBindDataset
from datasets.make_dataset import seed_worker

DEVICE='cuda:0'

def get_pdbbind_dataloader(cfg):
    dataset = PDBBindDataset(cfg)
    n_workers = cfg.platform.num_workers
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size,
                            num_workers=n_workers, pin_memory=True,
                            shuffle=False, worker_init_fn=seed_worker)
    return dataloader

def speed_test(cfg, model):
    loader = get_pdbbind_dataloader(cfg)
    all_batches = [ batch.to(DEVICE) for batch in tqdm(loader) ]
    t1 = time()
    for batch in tqdm(all_batches):
        model(batch, loader.dataset)
    t2 = time()
    diff = t2 - t1
    print(f"Took {diff} seconds")
    avg_time = diff/len(loader.dataset)
    print(f"Average time was {avg_time}")

def get_run_val_model(cfg, run_id, tag):
    api = wandb.Api()
    run = api.run(f"{cfg.project}/{run_id}")
    cfg = get_run_config(run, cfg)
    model = OldModel(cfg, run, tag).to(DEVICE)
    return model, cfg

if __name__ == "__main__":
    cfg = get_config()
    model, cfg = get_run_val_model(cfg, "37jstv82", "v4")
    with torch.no_grad():
        speed_test(cfg, model)
