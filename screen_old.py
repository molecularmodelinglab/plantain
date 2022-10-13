import sys
sys.path.insert(0, './terrace')

import pickle
import os
import random
import torch
import wandb
import pandas as pd
from glob import glob
from git_timewarp import GitTimeWarp
from tqdm import tqdm

from datasets.graphs.mol_graph import MolGraph, mol_graph_from_sdf
from datasets.graphs.prot_graph import ProtGraph, prot_graph_from_pdb
from datasets.data_types import ActivityData, IsActiveData
from terrace.batch import make_batch
from routines.ai_routine import AIRoutine
from common.cfg_utils import get_config, get_run_config

def get_name_to_model(cfg, run_ids, cache=False):
    print("Loading models...")
    name2model = {}
    cache_file = "artifacts/name2model.pkl"
    if cache:
        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except KeyboardInterrupt:
            raise
        except FileNotFoundError:
            pass
    api = wandb.Api()
    for run_id in run_ids:
        run = api.run(f"{cfg.project}/{run_id}")
        cfg = get_run_config(run, cfg)
        artifact = api.artifact(f"{cfg.project}/model-{run.id}:latest", type='model')
        artifact_dir = f"artifacts/model-{run_id}:{artifact.version}"
        if not os.path.exists(artifact_dir):
            assert artifact_dir == artifact.download()
        checkpoint_file = artifact_dir + "/model.ckpt"
        routine = AIRoutine.from_checkpoint(cfg, checkpoint_file)
        name2model[run.name] = routine.model

    with open(cache_file, "wb") as f:
        pickle.dump(name2model, f)

    return name2model

def get_batches(cfg, screen_df, cache=True):
    print(f"Getting batches for {screen_df.pocket[0]}")

    cache_folder = cfg.platform.cache_dir + "/val_screens/"
    cache_file = cache_folder + screen_df.pocket[0] + ".pkl"
    os.makedirs(cache_folder, exist_ok=True)

    if cache:
        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except KeyboardInterrupt:
            raise
        except FileNotFoundError:
            pass

    batches = []
    for i, row in tqdm(screen_df.iterrows(), total=len(screen_df)):
        # todo: will not be neccessary once the next V of BigBing comes out
        # if len(row.lig_smiles) < 5: continue
        lig_file = cfg.platform.bigbind_dir + "/" + row.lig_file
        rec_file = cfg.platform.bigbind_dir + "/" + row.ex_rec_pocket_file
        lig_graph = mol_graph_from_sdf(cfg, lig_file)
        rec_graph = prot_graph_from_pdb(cfg, rec_file)
        is_active = torch.tensor(row.active, dtype=bool)
        data = IsActiveData(lig_graph, rec_graph, is_active)
        batches.append(make_batch([data]))

    with open(cache_file, "wb") as f:
        pickle.dump(batches, f)

    return batches

def get_preds(cfg, name, model, batches, screen_df, cache=True):

    cache_folder = cfg.platform.cache_dir + "/screen_preds/" + screen_df.pocket[0] + "/"
    cache_file = cache_folder + name + ".pkl"
    os.makedirs(cache_folder, exist_ok=True)

    if cache:
        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except KeyboardInterrupt:
            raise
        except FileNotFoundError:
            pass

    preds = []
    with torch.no_grad():
        for batch in tqdm(batches):
            preds.append(model(batch))

    with open(cache_file, "wb") as f:
        pickle.dump(preds, f)

    return preds

def get_results(preds, batches):
    one_percent = round(len(batches)*0.01)
    pred_act = sorted(zip(preds, batches), key=lambda x: -x[0])[:one_percent]
    are_active = [ batch.is_active for _, batch in pred_act ]
    tot_actives = sum([batch.is_active for batch in batches])
    max_actives = min(tot_actives, one_percent)
    efi = sum(are_active)/len(are_active)
    nefi = efi*(one_percent/max_actives)
    return {
        "efi": efi.numpy()[0],
        "nefi": nefi.numpy()[0],
        "enrichment": (efi/(tot_actives/len(batches))).numpy()[0]
    }

def screen_single(cfg, name2model, screen_df):
    print(f"Screening {screen_df.pocket[0]}")
    batches = get_batches(cfg, screen_df)
    results = {}
    for name, model in name2model.items():
        print(f"Predicting with {name}")
        preds = get_preds(cfg, name, model, batches, screen_df)
        results[name] = get_results(preds, batches)
        print(f"  results for {name}: {results[name]}")
    
    rand_preds = [random.random() for batch in batches]
    results["rand"] = get_results(rand_preds, batches)

    return results

def results_to_df(results):
    rows = []
    for pocket, poc_res in results.items():
        row = { "pocket": pocket }
        for model_name, metrics in poc_res.items():
            for met_name, metric in metrics.items():
                row[f"{model_name}_{met_name}"] = metric
        rows.append(row)
    return pd.DataFrame(rows)

def save_results(results):
    os.makedirs("outputs", exist_ok=True)
    df = results_to_df(results)
    out_file = "outputs/val_screen_results.csv"
    print(out_file)
    df.to_csv(out_file, index=False)

def screen_all(cfg, run_ids):
    name2model = get_name_to_model(cfg, run_ids)
    results = {}
    for screen_csv in glob(cfg.platform.bigbind_dir + "/val_screens/*.csv"):
        max_pocket_size=42
        screen_df = pd.read_csv(screen_csv)
        screen_df = screen_df.query("num_pocket_residues >= 5 and pocket_size_x < @max_pocket_size and pocket_size_y < @max_pocket_size and pocket_size_z < @max_pocket_size").reset_index(drop=True)
        if len(screen_df) < 10: continue
        pocket = screen_df.pocket[0]
        results[pocket] = screen_single(cfg, name2model, screen_df)
    print("Saving results")
    save_results(results)

if __name__ == "__main__":
    cfg = get_config("./configs", "short_thicc_op_gnn")
    run_ids = ["1nhqz8vw", "1socj7qg" ]
    screen_all(cfg, run_ids)