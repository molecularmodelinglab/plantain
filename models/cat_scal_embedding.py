import warnings
import torch
import torch.nn as nn
from terrace import Module, LazyEmbedding, LazyLinear

class CatScalEmbedding(Module):
    """ For graphs with both categorical and scalar features,
    embed all cat. features and concatenate them with the scalar
    feats. """

    def __init__(self, embed_size):
        super().__init__()
        self.embed_size = embed_size

    def forward(self, batch):
        self.start_forward()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ret = []
            if batch.scal_feat.shape[-1] > 0:
                ret.append(self.make(LazyLinear, self.embed_size)(batch.scal_feat))
            if batch.cat_feat.shape[-1] > 0:
                ret.append(self.make(LazyEmbedding, self.embed_size)(batch.cat_feat))
        return torch.cat(ret, -1)