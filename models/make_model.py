from models.gnn_bind import GNNBind
from models.fp_nn import FpNN
from models.learnable_ff import LearnableFF
from models.outer_prod_gnn import OuterProdGNN
from models.interaction_gnn import InteractionGNN

name2model_cls = {
   "gnn_bind": GNNBind,
   "fp_nn": FpNN,
   "learnable_ff": LearnableFF,
   "outer_prod_gnn": OuterProdGNN,
   "interaction_gnn": InteractionGNN,
}

def get_model_cls(cfg):
    return name2model_cls[cfg.model.type]

def make_model(cfg, in_node,):
    mdl_cls = get_model_cls(cfg)
    return mdl_cls(cfg, in_node)