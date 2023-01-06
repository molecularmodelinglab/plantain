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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scal = self.make(LazyLinear, self.embed_size)(batch.scal_feat)
            cat = self.make(LazyEmbedding, self.embed_size)(batch.cat_feat)
        return torch.cat((scal, cat), -1)