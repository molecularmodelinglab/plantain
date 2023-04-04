from datetime import datetime
import sys
import cProfile
import traceback
import torch
from datasets.make_dataset import make_dataloader, make_dataset
from models.make_model import make_model
from common.cfg_utils import get_config
from terrace.batch import collate
from training.trainer import Trainer
from validation.metrics import get_metrics
from validation.validate import validate
from tqdm import tqdm, trange
from training.loss import get_losses
from common.wandb_utils import get_old_model

def profile(task):

    stamp = datetime.timestamp(datetime.now())
    out_fname = f"outputs/{task}-{stamp}.prof"

    if task == "inference":
        cfg = get_config("twister_v2")
        model = make_model(cfg)
        loader = make_dataloader(cfg, "bigbind_act", "val", model.get_input_feats())
        x, y = next(iter(loader))
        model(x)
        x = x.to("cuda")
        model = model.to("cuda")

        it = iter(loader)
        def fn():
            for i in trange(500):
                x, y = next(it)
                x = x.to("cuda")
                l = model(x).active_prob_unnorm
                l.backward()
                torch.zero_grad()
            torch.cuda.synchronize()
    elif task == "validate":
        cfg = get_config("twister_v2")
        cfg.batch_size=1
        model = make_model(cfg)
        model.cache_key = "dummy"
        # model = get_old_model(cfg, "intra_lig_energy", "best_k")
        fn = lambda: validate(cfg, model, cfg.val_datasets[0], "val", 30)
    elif task == "train":
        # cfg = get_config("attention_gnn")
        # cfg.profile_max_batches = 250
        # trainer = Trainer(cfg)
        # fn = lambda: trainer.fit(None, [])
        cfg = get_config("twister_v2")
        cfg.profile_max_batches = 50
        trainer = Trainer(cfg)
        fn = lambda: trainer.fit(None, [])
    elif task == "diff_train":
        cfg = get_config("diffusion")
        cfg.platform.batch_size = 8
        cfg.platform.num_workers = 0
        model = make_model(cfg)
        train_dataloader = make_dataloader(cfg, cfg.train_dataset, "train", model.get_input_feats())
        x, y = next(iter(train_dataloader))
        model(x)
        optim = torch.optim.AdamW(model.parameters(), lr=cfg.learn_rate)
        tasks = set(model.get_tasks()).intersection(train_dataloader.dataset.get_tasks())
        metrics = get_metrics(cfg, tasks)
        def train():
            for i, (x,y) in enumerate(tqdm(train_dataloader)):
                if i > 50:
                    break
                optim.zero_grad()
                pred = model.predict_train(x, y, tasks)
                loss, loss_dict = get_losses(cfg, tasks, x, pred, y)
                loss.backward()
                for key, val in metrics.items():
                    val.update(x, pred, y)
        fn = train
    else:
        raise ValueError(f"Invalid profile task {task}")

    pr = cProfile.Profile()
    pr.enable()
    try:
        fn()
    except:
        traceback.print_exc()
    pr.disable()
    print(f"Saving profile to {out_fname}")
    pr.dump_stats(out_fname)

if __name__ == "__main__":
    profile(sys.argv[1])