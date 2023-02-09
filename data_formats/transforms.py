
from dataclassy import dataclass
from typing import Any, Callable, List
from data_formats.graphs.mol_graph import get_mol_coords

from terrace import Batch, DFRow

@dataclass
class Transform:
    all_transforms = {}

    in_features: List[str]
    out_feature: str
    apply: Callable

    def __call__(self, x: DFRow) -> Any:
        return self.apply(x)

    @staticmethod
    def apply_many(transform_names, x: DFRow) -> DFRow:
        out = x.asdict()
        for name in transform_names:
            out[name] = Transform.all_transforms[name](x)
        return DFRow(**out)

def transform(in_features):
    def impl(func):
        ret = Transform(in_features, func.__name__, func)
        Transform.all_transforms[func.__name__] = ret
        return ret
    return impl

@transform(["lig"])
def lig_embed_pose(x):
    return get_mol_coords(x.lig, 0)

