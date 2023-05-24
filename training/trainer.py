import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.trainer.supporters import CombinedDataset
from common.utils import flatten_dict
from datasets.combo_dataloader import ComboDataloader
from terrace import collate
from git_timewarp import GitTimeWarp

from datasets.make_dataset import make_dataloader, make_train_dataloader
from models.make_model import make_model
from validation.metrics import get_metrics, reset_metrics
from .loss import get_losses

class Trainer(pl.LightningModule):
    """ General trainer for the neural networks """

    @classmethod
    def from_checkpoint(cls, cfg, checkpoint_file, commit=None):
        return cls.load_from_checkpoint(checkpoint_file, cfg=cfg, commit=commit)

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        from models.make_model import make_model
        self.model = make_model(cfg)

        self.metrics = {} # nn.ModuleDict()

        for name in self.cfg.val_datasets:
            val_loader = make_dataloader(self.cfg, name, "val", self.model.get_input_feats())

            # give the model an initial batch before training to initialize
            # its (lazily created) parameters
            x, y = collate([val_loader.dataset[0]])
            self.model(x)
    
    def get_tasks(self, prefix, dataset_idx):
        dataset = self.get_dataset(prefix, dataset_idx)
        return set(self.model.get_tasks()).intersection(dataset.get_tasks())

    @property
    def device(self):
        return next(iter(self.parameters())).device

    def make_metrics(self, name, tasks, dataset_idx):
        if dataset_idx is None:
            key = f"{name}_metrics"
        else:
            key = f"{name}_metrics_{dataset_idx}"
        if key not in self.metrics:
            self.metrics[key] = get_metrics(self.cfg, tasks).to(self.device)
        return self.metrics[key]

    def get_metrics(self, name):
        ret = nn.ModuleDict()
        for key in self.metrics:
            if f"{name}_metrics" in key:
                ret.update(self.metrics[key])
        return ret
        
    def get_dataloader(self, prefix):
        if prefix == 'train':
            return self.trainer.train_dataloader.loaders
        elif prefix == 'val':
            return self.trainer.val_dataloaders

    def get_dataset(self, prefix, dataset_idx):
        loader = self.get_dataloader(prefix)
        if isinstance(loader, list):
            if dataset_idx is None:
                dataset_idx = 0
            loader = loader[dataset_idx]

        if isinstance(loader, ComboDataloader) and dataset_idx is not None:
            return loader.loaders[dataset_idx].dataset

        if isinstance(loader.dataset, CombinedDataset):
            dataset = loader.dataset.datasets
        else:
            dataset = loader.dataset

        return dataset

    def shared_eval(self, prefix, batch, batch_idx, dataset_idx=None):
        print("")
        print("Evalling", prefix)
        print("")

        x, y = batch
        tasks = self.get_tasks(prefix, dataset_idx)
        metrics = self.make_metrics(prefix, tasks, dataset_idx)

        pred = self.model.predict_train(x, y, tasks, prefix, batch_idx)
        loss, loss_dict = get_losses(self.cfg, tasks, x, pred, y)

        if dataset_idx is not None:
            self.log(f"{prefix}/loss", loss, prog_bar=True, batch_size=len(x))
        for key, val in loss_dict.items():
            self.log(f"{prefix}/{key}", val, prog_bar=True, batch_size=len(x))
        
        on_step = prefix == "train"
        on_epoch = not on_step
        computed_metrics = {}
        for key, val in metrics.items():
            print(key)
            val.update(x, pred, y)
            if prefix == 'train' and batch_idx % self.cfg.metric_reset_interval == 0:
                computed_metrics[key] = val.compute()
                # val.apply(reset_metrics)

        dataset = self.get_dataset(prefix, dataset_idx)

        # if "batch_size" in self.cfg:
        #     tot_batches = len(dataset)//self.cfg.batch_size
        #     is_last_batch = (tot_batches == batch_idx + 1)
        # else:
        #     loader = self.get_dataloader(prefix)
        #     if isinstance(loader, list):
        #         if dataset_idx is None:
        #             dataset_idx = 0
        #         loader = loader[dataset_idx]
        #     is_last_batch = loader.batch_sampler.is_last_batch
        #     if is_last_batch:
        #         print("!!!!!!!!")
        #         print("LASSSSSS")
        #         print("!!!!!!!")

        dataset_name = dataset.get_name()
        for key, val in flatten_dict(computed_metrics).items():
            self.log(f"{prefix}/{dataset_name}/{key}", val, prog_bar=False, on_step=on_step, on_epoch=on_epoch, batch_size=len(x), add_dataloader_idx=False)
        
        # if is_last_batch and prefix != "train":
        #     self.log_all_metrics(prefix, dataset_idx)

        if "profile_max_batches" in self.cfg and batch_idx >= self.cfg.profile_max_batches:
            raise RuntimeError("Stop the process!")

        return loss

    def training_step(self, batch, batch_idx):
        loader = self.get_dataloader("train")
        if isinstance(loader, ComboDataloader):
            dataset_idx = loader.get_dataset_index(batch_idx)
            return self.shared_eval("train", batch, batch_idx, dataset_idx)
        return self.shared_eval('train', batch, batch_idx)
    
    def validation_step(self, batch, batch_idx, dataset_idx=None):
        print("!!!!")
        print("!!!!")
        return self.shared_eval('val', batch, batch_idx, dataset_idx)

    def test_step(self, batch, batch_idx, dataset_idx=None):
        return self.shared_eval('test', batch, batch_idx, dataset_idx)
    
    def on_validation_epoch_end(self):
        print("")
        print("Val end")
        # todo cant handle multiple dataloaders
        self.log_all_metrics("val", None)
        self.get_metrics("val").apply(reset_metrics)

    def log_all_metrics(self, prefix, dataset_idx):
        dataset_name = self.get_dataset(prefix, dataset_idx).get_name()
        tasks = self.get_tasks(prefix, dataset_idx)
        metrics = self.make_metrics(prefix, tasks, dataset_idx)
        computed_metrics = {}
        for key, val in metrics.items():
            computed_metrics[key] = val.compute()
        for key, val in flatten_dict(computed_metrics).items():
            self.log(f"{prefix}/{dataset_name}/{key}", val, prog_bar=False, on_epoch=True, batch_size=1, add_dataloader_idx=False)

    def on_train_end(self):
        pass
        # self.get_metrics("train").apply(reset_metrics)

    # def on_validation_end(self):
    #     self.get_metrics("val").apply(reset_metrics)

    def on_test_end(self):
        pass
        # self.get_metrics("test").apply(reset_metrics)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.cfg.learn_rate)

    def fit(self, logger, callbacks):

        train_loader = make_train_dataloader(self.cfg, self.model.get_input_feats())
        # train_loader = make_dataloader(self.cfg, self.cfg.train_dataset, "train", self.model.get_input_feats())
        val_loaders = []
        for name in self.cfg.val_datasets:
            val_loader = make_dataloader(self.cfg, name, "val", self.model.get_input_feats())
            val_loaders.append(val_loader)

        gpus = int(torch.cuda.is_available())

        # from pytorch_lightning.profiler import PyTorchProfiler
        # profiler = PyTorchProfiler()

        self.trainer = pl.Trainer(gpus=gpus,
                             num_sanity_val_steps=0,
                             max_epochs=self.cfg.max_epochs,
                             val_check_interval=self.cfg.val_check_interval,
                             check_val_every_n_epoch=self.cfg.get("check_val_every_n_epoch", 1),
                             log_every_n_steps=self.cfg.metric_reset_interval,
                             logger=logger,
                             callbacks=callbacks,
                              #profiler=profiler,
                             resume_from_checkpoint=None)

        self.trainer.fit(self, train_loader, val_loaders)
