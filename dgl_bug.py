import torch
import dgl
import pytorch_lightning as pl

class MyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10, 10)

    def training_step(self, batch, batch_nb):
        return self.layer(batch[1]).sum()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

class MyDataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        
    def __len__(self):
        return 10

    def __getitem__(self, idx):
        g = dgl.graph(data=([0,1],[1,0]), num_nodes=2)
        return g, torch.zeros((10,10), dtype=torch.float)

def collate_graphs(samples):
    graphs = [x[0] for x in samples]
    batched_graph = dgl.batch(graphs)
    targets = torch.cat([x[1] for x in samples])
    return batched_graph, targets

loader = torch.utils.data.DataLoader(dataset=MyDataset(), batch_size=2, num_workers=2, collate_fn=collate_graphs)
model = MyModel()

trainer = pl.Trainer(
    strategy='ddp',
    accelerator='gpu',
    devices=[0],
    fast_dev_run=True,
)

trainer.fit(model, loader)