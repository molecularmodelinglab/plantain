from collections import defaultdict
import os
import shutil
import subprocess
from rdkit.Chem.rdMolAlign import CalcRMS

from tqdm import tqdm
from common.cfg_utils import get_config
from common.utils import get_mol_from_file
from common.wandb_utils import get_old_model
from datasets.bigbind_struct import BigBindStructDataset
from vina_bigbind import prepare_rec

def mean_acc(rms_dict, thresh=2.0):
    return sum([ sum([ v < thresh for v in val ])/len(val) for val in rms_dict.values()])/len(rms_dict)

class DummyModel:

    def __init__(self, cache_key):
        self.cache_key = cache_key

def minimize_preds(cfg):
    # model = get_old_model(cfg, "even_more_tor", "best_k")
    model = DummyModel("wandb:zza5dhen:v4")

    pred_folder = f"outputs/pose_preds/{model.cache_key}/"
    out_folder = f"outputs/pose_preds/{model.cache_key}-minimized/"

    # shutil.rmtree(out_folder, ignore_errors=True)
    os.makedirs(out_folder, exist_ok=True)

    docked_min_rms = defaultdict(list)
    min_true_rms = defaultdict(list)
    docked_true_rms = defaultdict(list)

    acc_thresh = 2.0
    local_min_thresh = 2.0

    dataset = BigBindStructDataset(cfg, "val", [])
    for i, (rf, pocket) in enumerate(zip(tqdm(dataset.structures.ex_rec_file), dataset.structures.pocket)):

        # rf = cfg.platform.bigbind_dir + "/" + rf
        docked_sdf = pred_folder + f"/{i}.sdf"
        out_sdf = out_folder + f"/{i}.sdf"

        try:
            min_mol = get_mol_from_file(out_sdf)
        except OSError:
            # print("TODO: no more breaking")
            # continue
            rf = prepare_rec(cfg, rf)
            cmd = f"{cfg.platform.gnina_sif} gnina -r {rf} -l {docked_sdf} --minimize -o {out_sdf}"
            proc = subprocess.Popen(cmd.split(" "), stdout=subprocess.PIPE)
            out, err = proc.communicate()
            min_mol = get_mol_from_file(out_sdf)

        docked_mol = get_mol_from_file(docked_sdf)
        true_lig = get_mol_from_file(dataset.get_lig_crystal_file(i))

        docked_min_rms[pocket].append(CalcRMS(docked_mol, min_mol, maxMatches=1000))
        min_true_rms[pocket].append(CalcRMS(true_lig, min_mol, maxMatches=1000))
        docked_true_rms[pocket].append(CalcRMS(docked_mol, true_lig, maxMatches=1000))
        
    print(f"Plantain acc: ", mean_acc(docked_true_rms, acc_thresh))
    print(f"Minimized plantain acc: ", mean_acc(min_true_rms, acc_thresh))
    print(f"Frac of time plantain is in local min: ", mean_acc(docked_min_rms, local_min_thresh))

    local_min_rms = defaultdict(list)
    not_min_rms = defaultdict(list)

    min_local_min_rms = defaultdict(list)
    min_not_min_rms = defaultdict(list)
    for pocket in docked_min_rms.keys():
        for min_rms, true_rms, mt_rms in zip(docked_min_rms[pocket], docked_true_rms[pocket], min_true_rms[pocket]):
            if min_rms < local_min_thresh:
                local_min_rms[pocket].append(true_rms)
                min_local_min_rms[pocket].append(mt_rms)
            else:
                not_min_rms[pocket].append(true_rms)
                min_not_min_rms[pocket].append(mt_rms)


    print(f"Acc when in local min: ", mean_acc(local_min_rms, acc_thresh))
    print(f"Acc when not in local min: ", mean_acc(not_min_rms, acc_thresh))

    print(f"Minimized acc when in local min: ", mean_acc(min_local_min_rms, acc_thresh))
    print(f"Minimized acc when not in local min: ", mean_acc(min_not_min_rms, acc_thresh))

if __name__ == "__main__":
    cfg = get_config("diffusion_v2")
    minimize_preds(cfg)