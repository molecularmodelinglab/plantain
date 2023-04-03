import torch
from tqdm import trange
from datasets.make_dataset import make_dataloader, make_dataset
from models.make_model import make_model
from common.cfg_utils import get_config
from terrace.batch import collate
from models.twister_v2 import TwistBlock, TwistEncoder

if __name__ == "__main__":
    cfg = get_config("twister_v2")
    model = make_model(cfg)
    loader = make_dataloader(cfg, "bigbind_act", "val", model.get_input_feats())
    x, y = next(iter(loader))
    enc = TwistEncoder(cfg.model)
    block = TwistBlock(cfg.model)
    td = enc(x)
    res_index = x.full_rec_data.get_res_index()
    block.update_simple(x, td, res_index)

    x = x.to("cuda")
    enc = enc.to("cuda")
    block = block.to("cuda")
    res_index = res_index.to("cuda")

    td = enc(x)

    for i in trange(50000):
        # td = enc(x)
        block.update_simple(x, td, res_index)