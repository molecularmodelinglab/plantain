import sys
sys.path.insert(0, './terrace')

import wandb
import torch
from tqdm import tqdm
from torchmetrics import ROC, AUROC
import matplotlib.pyplot as plt

from terrace.batch import DataLoader

from common.metrics import get_metrics
from datasets.make_dataset import make_dataloader, seed_worker
from datasets.bigbind_screen import BigBindScreenDataset
from common.old_routine import get_old_model, old_model_key
from common.cfg_utils import get_config, get_run_config
from common.cache import cache
from common.plot_metrics import plot_metrics

if __name__ == "__main__":
    cfg = get_config()
    split = "val"
    for target in BigBindScreenDataset.get_all_targets(cfg, split):
        print(f"Loading all screening data for {target}")
        dataset = BigBindScreenDataset(cfg, target, split)
        n_workers = cfg.platform.num_workers
        loader = DataLoader(dataset, batch_size=cfg.batch_size,
                                num_workers=n_workers, pin_memory=True,
                                shuffle=False, worker_init_fn=seed_worker)
        for batch in tqdm(loader):
            pass
    # for sna_frac in (None, 1):
    #     for use_rec in (True, False):
    #         cfg.data.sna_frac = sna_frac
    #         cfg.data.use_rec = use_rec
    #         print(f"Loading all the data with {sna_frac=}, {use_rec=}, {split=}")
    #         loader = make_dataloader(cfg, split)
    #         for batch in tqdm(loader):
    #             pass