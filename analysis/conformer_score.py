from tqdm import tqdm
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import rdkit
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolAlign
from common.utils import *
from common.metrics import *
from common.losses import *
from common.cfg_utils import *
from common.old_routine import *
from datasets.make_dataset import *
from models.gnn_bind import *
from models.learnable_ff import LearnableFF
from routines.ai_routine import *
from terrace.batch import *
from terrace.type_data import *
from terrace.comp_node import *
from torchmetrics import *
from datasets.data_types import *
from models.val_model import *
from datasets.inference_dataset import InferenceDataset
from datasets.bigbind_screen import *
from datasets.pdbbind import *
from datasets.lit_pcba import *
from datasets.graphs.interaction_graph import *
from datasets.graphs.plot_graph import *
import matplotlib.pyplot as plt
from traceback import print_exc
import subprocess
import seaborn as sns
import torch
from common.cache import *
from glob import glob
import wandb
from git_timewarp import GitTimeWarp
from meeko import *
from datasets.bigbind_vina import *
from models.interaction_gnn import *

@cache(lambda *args: None)
def get_all_rmsds(cfg, df):
    all_rmsds = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        docked_file = cfg.platform.bigbind_vina_dir + "/" + row.docked_lig_file
        lig_file = cfg.platform.bigbind_dir + "/" + row.lig_file
        lig = get_mol_from_file(lig_file)
        docked_lig = get_mol_from_file(docked_file)
        
        rmsds = []
        for conf in range(docked_lig.GetNumConformers()):
            rmsds.append(rdMolAlign.CalcRMS(lig, docked_lig, 0, conf))
        all_rmsds.append(rmsds)
    return all_rmsds

def get_vina_acc(cfg, all_rmsds, cutoff=2):

    all_correct = []
    for rmsds in all_rmsds:
        all_correct.append([ rmsd < cutoff for rmsd in rmsds ])

    # %%
    top_n_acc = []
    for n in range(16):
        top_n_acc.append(sum([ True in correct[:n] for correct in all_correct])/len(all_rmsds))

    # %%
    for n, acc in enumerate(top_n_acc):
        if n == 0: continue
        print(f"Top {n} acc: {acc:.2f}")


@cache(old_model_key)
def get_all_conf_scores(cfg, run, tag):
    device='cuda:0'
    model = get_old_model(cfg, run, tag)
    model.eval()
    model = model.to(device)

    all_scores = []
    for i, row in tqdm(df.iterrows(), total=len(df)):

        docked_file = cfg.platform.bigbind_vina_dir + "/" + row.docked_lig_file
        rec_file = cfg.platform.bigbind_dir + "/" + row.ex_rec_pocket_file
        lig_file = cfg.platform.bigbind_dir + "/" + row.lig_file
        lig = get_mol_from_file(lig_file)
        lig = Chem.RemoveHs(lig)
        rec = get_prot_from_file(rec_file)
        docked_lig = get_mol_from_file(docked_file)
        docked_lig = Chem.RemoveHs(docked_lig)

        # %%
        rec_graph = ProtGraph(cfg, rec)

        inter_graphs = []
        for conformer in range(docked_lig.GetNumConformers()):
            lig_graph = MolGraph(cfg, docked_lig, conformer)
            inter_graph = InteractionGraph(cfg, lig_graph, rec_graph)
            inter_graphs.append(inter_graph)

        is_active = torch.tensor(False, dtype=bool)
        batch = make_batch([InteractionActivityData(tuple(inter_graphs), is_active)]).to(device)

        # %%
        scores = model.get_conformer_scores(batch)
        all_scores.append(scores.tolist())

    return all_scores


if __name__ == "__main__":
    cfg = get_config("vina")
    df = pd.read_csv(cfg.platform.bigbind_vina_dir + "/structures_val.csv")
    all_rmsds = get_all_rmsds(cfg, df)
    # get_vina_acc(cfg, all_rmsds)
    run_id = "122thu0a"
    api = wandb.Api()
    run = api.run(f"{cfg.project}/{run_id}")
    with torch.no_grad():
        all_scores = get_all_conf_scores(cfg, run, "latest")
    


