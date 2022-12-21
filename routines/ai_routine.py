import pytorch_lightning as pl
import torch
import torch.nn as nn

from terrace.comp_node import Input
from common.cfg_utils import get_all_tasks
from common.task_metrics import get_epoch_task_metrics, get_task_metrics
from models.make_model import make_model
from datasets.make_dataset import make_dataloader
from common.losses import get_losses
from common.metrics import get_metrics
from common.plot_metrics import plot_metrics

class AIRoutine(pl.LightningModule):
    """ A "Routine" is basically just a lightning module -- something that's model
    agnostic, but defines how the training and eval is performed for a particular
    type of model (e.g. activitity prediction) """

    # todo: this whole class is bad bacause when you save the checkpoints it saves the datasets
    # as well, which are pretty big. I thought I was being clever with this whole Routine thing
    # but will need to refactor later

    @classmethod
    def from_checkpoint(cls, cfg, checkpoint_file):
        return cls.load_from_checkpoint(checkpoint_file, cfg=cfg, checkpoint_file=checkpoint_file)

    def __init__(self, cfg, checkpoint_file = None):
        super().__init__()
        self.checkpoint_file = checkpoint_file
        self.cfg = cfg
        self.learn_rate = cfg.learn_rate
        self.train_dataloader = make_dataloader(cfg, "train")
        self.val_dataloader = make_dataloader(cfg, "val")
        self.metrics = nn.ModuleDict({
            "train_metric": get_metrics(cfg),
            "val_metric": get_metrics(cfg)
        })

        batch = next(iter(self.val_dataloader))

        self.val_variance = self.val_dataloader.dataset.get_variance()
        in_node = Input(self.val_dataloader.get_type_data())
        self.model = make_model(cfg, in_node)

    def forward(self, batch):
        act = self.model(batch)
        return act

    def shared_eval(self, batch, batch_idx, prefix):
        pred = self(batch)
        loss, loss_dict = get_losses(self.cfg, batch, pred)
        self.log(f"{prefix}_loss", loss, prog_bar=True, batch_size=len(batch))
        for key, val in loss_dict.items():
            self.log(f"{prefix}_{key}", val, prog_bar=True, batch_size=len(batch))

        metrics = self.metrics[prefix + "_metric"]
        on_step = prefix == "train"
        on_epoch = not on_step
        for key, val in metrics.items():
            # debug: is AUROC causing the OOMs?
            if prefix == "train" and key in ("auroc", "roc", "r2", "mse"):
                # print("Skipping")
                continue
            val(pred, batch)
            if isinstance(val.compute(), torch.Tensor):
                self.log(f"{prefix}_{key}", val, prog_bar=False, on_step=on_step, on_epoch=on_epoch, batch_size=len(batch))
        
        task_mets = {}
        for task in get_all_tasks(self.cfg):
            task_mets.update(get_task_metrics(task, prefix, batch_idx, self.model, batch))
        for key, val in task_mets.items():
            self.log(f"{prefix}_{key}", val, prog_bar=True, on_step=on_step, on_epoch=on_epoch, batch_size=len(batch))
        
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_eval(batch, batch_idx, 'train')
    
    def validation_step(self, batch, batch_idx):
        return self.shared_eval(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self.shared_eval(batch, batch_idx, 'test')

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learn_rate)

    def on_validation_epoch_end(self) -> None:
        num_iter = 10 if self.trainer.state.stage == "sanity_check" else None
        metrics = self.get_all_epoch_task_metrics("val", num_iter)
        for key, val in metrics.items():
            self.log(f"val_{key}", val)

    def get_all_epoch_task_metrics(self, split, num_iter):
        ret = {}
        for task in get_all_tasks(self.cfg):
            ret.update(get_epoch_task_metrics(self.cfg, task, self.model, split, num_iter))
        return ret

    def fit(self, logger, callbacks):
        gpus = int(torch.cuda.is_available()) # self.cfg.gpus

        self.trainer = pl.Trainer(gpus=gpus,
                             max_epochs=self.cfg.max_epochs,
                             val_check_interval=self.cfg.val_check_interval,
                             logger=logger,
                             callbacks=callbacks,
                             # replace_sampler_ddp=False,
                             # strategy='ddp',
                             resume_from_checkpoint=self.checkpoint_file)

        self.trainer.fit(self, self.train_dataloader, self.val_dataloader)
        self.trainer.validate(self, self.val_dataloader)