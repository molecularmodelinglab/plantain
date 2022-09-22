from models.gnn_bind import GNNBind
from models.fp_nn import FpNN

name2model_cls = {
   "gnn_bind": GNNBind,
   "fp_nn": FpNN
}

def get_model_cls(cfg):
    return name2model_cls[cfg.model.type]

def make_model(cfg, in_node,):
    mdl_cls = get_model_cls(cfg)
    return mdl_cls(cfg, in_node)