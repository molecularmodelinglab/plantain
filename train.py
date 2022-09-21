from routines.routine import Routine
from common.cfg_utils import get_config

def train(cfg):
    routine = Routine(cfg)
    routine.fit(None, [], None)

if __name__ == "__main__":
    cfg = get_config()
    train(cfg)