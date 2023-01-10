import importlib
from models.model import Model

# import all the files in the directory so we can create
# a name to model mapping
import os
for module in os.listdir(os.path.dirname(__file__)):
    if module == '__init__.py' or module[-3:] != '.py':
        continue
    importlib.import_module('models.'+module[:-3])

def flatten_subclasses(cls):
    ret = [ cls ]
    for subclass in cls.__subclasses__():
        ret += flatten_subclasses(subclass)
    return ret

name_to_model_cls = {}
for class_ in flatten_subclasses(Model):
    try:
        name_to_model_cls[class_.get_name()] = class_
    except NotImplementedError:
        pass

def make_model(cfg):
    mdl_cls = name_to_model_cls[cfg.model.type]
    return mdl_cls(cfg)