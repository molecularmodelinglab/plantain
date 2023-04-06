from omegaconf import DictConfig
import torch
from models.force_field import rbf_encode

from terrace import Batch, CategoricalTensor
from .graph3d import Graph3d, Node3d, Edge3d

class DistEdge(Edge3d):

    @staticmethod
    def make_from_dists(cfg, dists):
        dists = torch.tensor(dists, dtype=torch.float32)
        if "edge_rbf" in cfg.data.rec_graph:
            rbf_cfg = cfg.data.rec_graph.edge_rbf
            scal_feat = rbf_encode(dists, rbf_cfg.start, rbf_cfg.stop, rbf_cfg.step)
        else:
            scal_feat = dists.unsqueeze(-1)
        cat_feat = torch.zeros((len(scal_feat), 0), dtype=torch.long)
        cat_feat = CategoricalTensor(cat_feat, num_classes=[])
        edata = Batch(DistEdge,
                      cat_feat=cat_feat,
                      scal_feat=scal_feat)
        return edata

    def __init__(self, prot_cfg: DictConfig, node1: Node3d, node2: Node3d):
        cat_feat = []
        scal_feat = []
        scal_feat.append(torch.linalg.norm(node1.coord - node2.coord))
        # for feat_name in prot_cfg.edge_feats:
        #     if feat_name == "dist":
        #         scal_feat.append(torch.linalg.norm(node1.coord - node2.coord))
        #     else:
        #         raise AssertionError()
        cat_feat = torch.tensor(cat_feat, dtype=torch.long)
        scal_feat = torch.tensor(scal_feat, dtype=torch.float32)
        super(DistEdge, self).__init__(cat_feat, scal_feat)