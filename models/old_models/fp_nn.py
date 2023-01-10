from terrace.make_module import MakeLinear, MakeBatchNorm1d
from terrace.typed_module import ReLU, Model

class FpNN(Model):

    def __init__(self, cfg, in_node):
        x = in_node[0]
        for sz in cfg.model.hidden_sizes:
            x = MakeLinear(sz)(x)
            x = MakeBatchNorm1d()(x)
            x = ReLU()(x)
        x = MakeLinear(1)(x)
        out = x[:,0]
        super(FpNN, self).__init__(in_node, out)