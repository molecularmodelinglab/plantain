import torch
from tqdm import tqdm
from common.cfg_utils import get_config
from common.wandb_utils import get_old_model
from models.gnina import GninaPose
from validation.metrics import get_rmsds
from validation.validate import get_preds
import matplotlib.pyplot as plt

def main(cfg, model_name="even_more_tor"):

    num_preds = None
    shuffle_val = False
    dataset_name = "bigbind_struct"
    subset_indexes = None

    print("Loading model")

    if model_name == "gnina":
        cfg.data.num_poses = 9
        model = GninaPose(cfg)
    else:
        model = get_old_model(cfg, model_name)

    print("Getting predictions")
    x, y, pred = get_preds(cfg, model, dataset_name, "val", num_preds, shuffle_val)

    print("Getting RMSDs")
    if model_name == "gnina":
        best_energies = -pred.pose_scores[:,0].cpu()
    else:
        best_energies = pred.energy[:,0].cpu()
    
    rmsds = get_rmsds(x.lig, pred.lig_pose.get(0), y.lig_crystal_pose).cpu()

    out_fname = 'outputs/pred_rmsd_acc.png'
    print(f"Creating figure and saving to {out_fname}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(best_energies, rmsds)
    ax.set_xlabel('Predicted RMSD')
    ax.set_ylabel('RMSD')

    plt.savefig(out_fname, dpi=100)


    print("Calculating selective accuracy")

    cutoff = 2.0
    energy_indexes = torch.argsort(best_energies)
    correct = rmsds < cutoff

    resolution = 100
    reject_fracs = torch.linspace(0.0, 0.9, resolution)
    final_reject_fracs = []
    accuracies = []
    for frac in tqdm(reject_fracs):
        frac = float(frac)
        to_remove = round(frac*len(x))
        if to_remove == 0:
            indexes = energy_indexes
        else:
            indexes = energy_indexes[:-to_remove]
        acc = float(correct[indexes].sum() / len(indexes))
        final_reject_fracs.append(frac)
        accuracies.append(acc)

    out_fname = "outputs/selective_accuracy.png"
    print(f"Creating figure and saving to {out_fname}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(final_reject_fracs, accuracies)
    ax.set_xlabel('Rejected fraction')
    ax.set_ylabel('Accuracy')

    plt.savefig(out_fname, dpi=100)


if __name__ == "__main__":
    cfg = get_config("diffusion_v2")
    main(cfg)