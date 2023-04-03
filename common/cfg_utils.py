from typing import MutableMapping
from omegaconf import OmegaConf
from torch.utils._pytree import tree_map

# OmegaConf access methods take too long. This speeds things up
def to_attr_dict(d):
    if isinstance(d, dict):
        return AttrDict({key: to_attr_dict(val) for key, val in d.items() })
    elif isinstance(d, list):
        return [ to_attr_dict(val) for val in d ]
    return d

class AttrDict(MutableMapping):
    def __init__(self, d):
        self.__dict__ = d

    def __getitem__(self, key):
        return self.__dict__[key]

    def get(self, key, default):
        return self.__dict__.get(key, default)

    def __setitem__(self, key, val):
        self.__dict__[key] = val

    def __delitem__(self, key):
        del self.__dict__[key]

    def __len__(self):
        return len(self.__dict__)

    def __contains__(self, item):
        return item in self.__dict__

    def __iter__(self):
        return iter(self.__dict__)

    def __next__(self):
        return next(self.__dict__)

    def __repr__(self) -> str:
        return f"AttrDict({repr(self.__dict__)})"

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

def get_config_dict(run):
    """ wandb flattens dicts when uploading hyperparams. This unflattens them """
    ret = {}
    for key, val in run.config.items():
        container = ret
        subkeys = key.split("/")
        for subkey in subkeys[:-1]:
            if subkey not in container:
                container[subkey] = {}
            container = container[subkey]
        if val == 'None':
            val = None
        container[subkeys[-1]] = val
    return ret

def get_run_config(run, cfg=None):
    """ Get the config used by the wandb run """
    ret = to_attr_dict(get_config_dict(run))
    if cfg is not None:
        ret.platform = cfg.platform
    return ret

def get_config(cfg_name, folder="./configs"):
    """ Loads the config file and merges in the platform cfg """
    base_conf = OmegaConf.load(folder + f"/{cfg_name}.yaml")
    platform_conf = OmegaConf.load(folder + "/local.yaml")
    cli_conf = OmegaConf.from_cli()
    cfg = OmegaConf.merge(base_conf, platform_conf, cli_conf)
    if cfg.project is not None and "project_postfix" in cfg:
        cfg.project = cfg.project + "_" + cfg.project_postfix
    return to_attr_dict(OmegaConf.to_container(cfg))

def get_all_tasks(cfg):
    """ Returns all the config tasks (if there is only one, creates
    a new list containing only the current task) """
    if isinstance(cfg.task, str):
        return [ cfg.task ]
    else:
        return cfg.task
