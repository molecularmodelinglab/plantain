import sys
from tqdm import tqdm
from common.cfg_utils import get_config
from datasets.make_dataset import make_dataloader


def dataset_iterate(cfg):
    train_dataloader = make_dataloader(cfg, "val")
    for i, data in enumerate(tqdm(train_dataloader)):
        if i > 100:
            break
        pass

if __name__ == "__main__":
    if len(sys.argv) > 1 and '=' not in sys.argv[1]:
        cfg_name = sys.argv[1]
    else:
        raise AssertionError("First argument must be name of config profile!")
    cfg = get_config(cfg_name=cfg_name)
    dataset_iterate(cfg)
