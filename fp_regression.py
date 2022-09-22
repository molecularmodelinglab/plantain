import sys
sys.path.insert(0, './terrace')

from tqdm import tqdm
import torch
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.metrics import r2_score

from datasets.make_dataset import make_dataloader
from common.cfg_utils import get_config

def get_entire_dataset(loader, max_size=None):
    X = []
    Y = []
    i = 0
    for x, y in tqdm(loader.dataset):
        X.append(x)
        Y.append(y)
        if max_size is not None and i > max_size: break
        i += 1
    return torch.stack(X).numpy(), torch.stack(Y).numpy()

def main(cfg):
    # train_loader = make_dataloader(cfg, "train")
    val_loader = make_dataloader(cfg, "val")
    # tX, tY = get_entire_dataset(train_loader)
    vX, vY = get_entire_dataset(val_loader)

    print(vY.mean(), vY.var())
    print(vX.shape, vY.shape)

    model = LinearRegression()
    model.fit(vX, vY)
    pY = model.predict(vX)
    print(pY[0:10])
    r2 = r2_score(vY, pY)
    print(f"R2 of model is: {r2}")

if __name__ == "__main__":
    cfg = get_config("./configs", "fp_regression")
    main(cfg)