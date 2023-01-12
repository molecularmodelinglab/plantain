import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.trainer.supporters import CombinedDataset
from common.utils import flatten_dict
from terrace import collate

from datasets.make_dataset import make_dataloader
from models.make_model import make_model
from validation.metrics import get_metrics, reset_metrics
from .loss import get_losses

class Trainer(pl.LightningModule):
    """ General trainer for the neural networks """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = make_model(cfg)
        self.metrics = nn.ModuleDict()
    
    def get_tasks(self, loader):
        if isinstance(loader.dataset, CombinedDataset):
            dataset = loader.dataset.datasets
            # dataset = loader.dataset.datasets[0]
        else:
            dataset = loader.dataset
        return self.model.get_tasks().intersection(dataset.get_tasks())

    @property
    def device(self):
        return next(iter(self.parameters())).device

    def make_metrics(self, name, tasks):
        key = f"{name}_metrics"
        if key not in self.metrics:
            self.metrics[key] = get_metrics(tasks).to(self.device)
        return self.metrics[key]

    def get_metrics(self, name):
        return self.metrics[f"{name}_metrics"]
        
    def get_dataloader(self, prefix):
        if prefix == 'train':
            return self.trainer.train_dataloader
        elif prefix == 'val':
            return self.trainer.val_dataloaders[0]

    def shared_eval(self, prefix, batch, batch_idx):
        x, y = batch
        tasks = self.get_tasks(self.get_dataloader(prefix))
        metrics = self.make_metrics(prefix, tasks)

        pred = self.model.predict(tasks, x)
        loss, loss_dict = get_losses(self.cfg, pred, y)

        self.log(f"{prefix}_loss", loss, prog_bar=True, batch_size=len(x))
        for key, val in loss_dict.items():
            self.log(f"{prefix}_{key}", val, prog_bar=True, batch_size=len(x))
        
        on_step = prefix == "train"
        on_epoch = not on_step
        computed_metrics = {}
        for key, val in metrics.items():
            val.update(x, pred, y)
            if prefix == 'train' and batch_idx % self.cfg.metric_reset_interval == 0:
                computed_metrics[key] = val.compute()
                val.apply(reset_metrics)

        for key, val in flatten_dict(computed_metrics).items():
            self.log(f"{prefix}_{key}", val, prog_bar=False, on_step=on_step, on_epoch=on_epoch, batch_size=len(x))
        
        if self.trainer.is_last_batch and prefix != "train":
            self.log_all_metrics(prefix)

        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_eval('train', batch, batch_idx)
    
    def validation_step(self, batch, batch_idx):
        return self.shared_eval('val', batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.shared_eval('test', batch, batch_idx)

    def log_all_metrics(self, prefix):
        tasks = self.get_tasks(self.get_dataloader(prefix))
        metrics = self.make_metrics(prefix, tasks)
        computed_metrics = {}
        for key, val in metrics.items():
            computed_metrics[key] = val.compute()
        for key, val in flatten_dict(computed_metrics).items():
            self.log(f"{prefix}_{key}", val, prog_bar=False, on_epoch=True, batch_size=1)

    def on_train_end(self):
        self.get_metrics("train").apply(reset_metrics)

    def on_validation_end(self):
        self.get_metrics("val").apply(reset_metrics)

    def on_test_end(self):
        self.get_metrics("test").apply(reset_metrics)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.cfg.learn_rate)

    def fit(self, logger, callbacks):

        train_loader = make_dataloader(self.cfg, self.cfg.train_dataset, "train", self.model.get_data_format())
        val_loader = make_dataloader(self.cfg, self.cfg.val_datasets[0], "val", self.model.get_data_format())

        # give the model an initial batch before training to initialize
        # its (lazily created) parameters
        x, y = collate([train_loader.dataset[0]])
        self.model(x)

        gpus = int(torch.cuda.is_available())

        self.trainer = pl.Trainer(gpus=gpus,
                             max_epochs=self.cfg.max_epochs,
                             val_check_interval=self.cfg.val_check_interval,
                             logger=logger,
                             callbacks=callbacks,
                             resume_from_checkpoint=None)

        self.trainer.fit(self, train_loader, val_loader)