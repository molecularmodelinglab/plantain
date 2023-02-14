from omegaconf import OmegaConf

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
    ret = OmegaConf.create(get_config_dict(run))
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
    return cfg

def get_all_tasks(cfg):
    """ Returns all the config tasks (if there is only one, creates
    a new list containing only the current task) """
    if isinstance(cfg.task, str):
        return [ cfg.task ]
    else:
        return cfg.task
