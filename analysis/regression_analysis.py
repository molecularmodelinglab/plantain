from common.cfg_utils import get_config
from validation.validate import validate

def validate_regression(cfg):
    print("Validating Ligand and Receptor on test set")
    validate(cfg, "34ednh2q", "v4", "test")
    print("Validating Ligand only on test set")
    validate(cfg, "21mnmh68", "v4", "test")


    print("Validating Ligand and Receptor on train set")
    validate(cfg, "34ednh2q", "v4", "train")
    print("Validating Ligand only on train set")
    validate(cfg, "21mnmh68", "v4", "train")

if __name__ == "__main__":
    cfg = get_config()
    validate_regression(cfg)