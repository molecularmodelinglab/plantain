import torch
import torch.nn as nn

class CatScalEmbedding(nn.Module):
    """ For graphs with both categorical and scalar features,
    embed all cat. features and concatenate them with the scalar
    feats. """

    def __init__(self, embed_size, td):
        super().__init__()
        self.embeddings = nn.ModuleList()
        total_dim = td.scal_feat.shape[-1]
        for i, val in enumerate(td.cat_feat.max_values):
            embedding = nn.Embedding(val, embed_size)
            total_dim += embed_size
            self.embeddings.append(embedding)
        self.total_dim = total_dim

    def forward(self, batch):
        ret = [ batch.scal_feat ]
        for i, embed in enumerate(self.embeddings):
            ret.append(embed(batch.cat_feat[:,i]))
        return torch.cat(ret, -1)