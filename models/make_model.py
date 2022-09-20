from models.gnn_bind import GNNBind

def make_model(cfg):
    return {
        "gnn_bind": GNNBind
    }[cfg.model.type](cfg)