import torch.nn as nn
import torch
import torch.nn.functional as F
from terrace import Batch, collate
from .model import Model
from .attention_gnn import AttentionGNN

class AttentionGNNPose(nn.Module, Model):

    def __init__(self, cfg, model: AttentionGNN):
        super().__init__()
        self.model = model

    @staticmethod
    def get_name() -> str:
        return "attention_gnn_pose"

    def get_data_format(self):
        return LigAndRecGraphMultiPose.make

    def forward(self, x):
        type_ = Data.create_type([x.item_type(), LigAndRec])
        batch = Batch(type_, lig_graph=collate([ lg[0] for lg in x.lig_graphs ]), **x.asdict())
        return self.model.forward(batch)[1]

    def get_tasks(self):
        return [ ScorePose ]

    def score_pose(self, x, pred):
        type_ = self.get_pred_type()

        scores = []
        for x0, pred0 in zip(x, pred):
            rec_coord = x0.rec_graph.ndata.coord
            cur_scores = []
            for lig_graph in x0.lig_graphs:
                lig_coord = lig_graph.ndata.coord
                dist = torch.cdist(lig_coord, rec_coord)

                nulls = torch.tensor([[5.0]]*len(lig_coord), device=lig_coord.device)
                dist = torch.cat((dist, nulls), 1)
                labels = torch.argmin(dist, 1)

                cur_scores.append(F.cross_entropy(pred0, labels))
            scores.append(-torch.stack(cur_scores))

        return Batch(type_, pose_scores=scores)


    