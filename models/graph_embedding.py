from terrace import Module, GraphBatch

from models.cat_scal_embedding import CatScalEmbedding

class GraphEmbedding(Module):
    """ Uses CatScalEmbedding to create hidden features for
    both the nodes and edges of the graph. Returns a tuple
    (node_feats, edge_feats) """

    def __init__(self, enc_cfg):
        super().__init__()
        self.node_sz = enc_cfg.node_embed_size
        self.edge_sz = enc_cfg.edge_embed_size

    def forward(self, graph: GraphBatch):
        self.start_forward()
        node_feats = self.make(CatScalEmbedding, self.node_sz)(graph.ndata)
        edge_feats = self.make(CatScalEmbedding, self.edge_sz)(graph.edata)
        return node_feats, edge_feats