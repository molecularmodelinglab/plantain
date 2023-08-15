
import torch
from common.cfg_utils import get_config
from datasets.inference_dataset import DummyDataset
from models.plantain import Plantain
from terrace.batch import collate


def get_pretrained_plantain():
    """ Creates the model from the paper and loads pretrained weights """
    cfg = get_config("icml")
    model = Plantain(cfg)
    dataset = DummyDataset(cfg, ['lig_embed_pose', 'lig_torsion_data', 'lig_graph', 'rec_graph', 'full_rec_data'])

    # because the submodules are lazily created, we must run this model
    # before loading the state dict
    x, y = dataset[0]
    model.get_hidden_feat(collate([x]))

    model.force_field.load_state_dict(torch.load("data/plantain_final.pt"))

    model.cache_key = "plantain"

    return model