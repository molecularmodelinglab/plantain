from dataclasses import dataclass
from typing import List

class Task:
    """ Note: _all_ tasks should be defined in this file. This is required
    for Datasets to know which tasks they qualify for (they loop over all
    the instances of Task) """

    all_tasks = {}

    name: str
    input_feats: List[str]
    label_feats: List[str]
    pred_feats: List[str]

    def __init__(self, name, input_feats, label_feats, pred_feats):
        self.name = name
        self.input_feats = input_feats
        self.label_feats = label_feats
        self.pred_feats = pred_feats
        Task.all_tasks[self.name] = self

Task("score_activity_class",
    ["lig", "rec"],
    ["is_active"],
    ["is_active_score"])
Task("score_activity_regr",
    ["lig", "rec"],
    ["activity"],
    ["activity_score"])
Task("classify_activity",
    ["lig", "rec"],
    ["is_active"],
    ["active_prob_unnorm", "active_prob"])
Task("predict_activity",
    ["lig", "rec"],
    ["activity"],
    ["activity"])
Task("reject_option",
    [],
    [],
    ["select_score"])
Task("predict_lig_pose",
    ["lig", "rec"],
    ["lig_crystal_pose"],
    ["lig_pose"])
Task("score_pose",
    ["rec","lig", "lig_docked_poses"],
    ["pose_rmsds"],
    ["pose_scores"])