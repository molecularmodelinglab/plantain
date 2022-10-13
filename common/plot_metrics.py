import wandb

def plot_metrics(metrics, split, to_wandb=False, run=None):
    if to_wandb:
        assert wandb.run is not None
        epoch = wandb.run.summary.epoch
        roc = metrics["roc"]
        fpr, tpr, thresh = roc
        data = [[x, y] for (x, y) in zip(fpr, tpr)]
        table = wandb.Table(data=data, columns = ["fpr", "tpr"])
        wandb.log({f"{split}_{epoch}_roc" : wandb.plot.line(table, "fpr", "tpr",
           title=f"{split} ROC for epoch {epoch}")})
    else:
        import matplotlib.pyplot as plt
        roc = metrics["roc"]
        fpr, tpr, thresh = roc
        assert run is not None
        plt.plot(fpr.cpu(), tpr.cpu(), label=run.name)
        plt.plot([0, 1], [0, 1], color='black')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.axis('equal')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title("ROC")
        plt.legend()
        plt.show()