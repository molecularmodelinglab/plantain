import os
import time
from traceback import print_exc

import pandas as pd
from tqdm import tqdm

from common.cfg_utils import get_config
from vina_bigbind import get_bigbind_dir, prepare_rec, run_vina

def get_lig_file(row):
    return row.lig_file

def get_rec_file(row):
    return row.ex_rec_file

def benchmark(cfg, program):
    df = pd.read_csv("outputs/bb_test_subset.csv")
    runtimes = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        try:
            lig_file = get_bigbind_dir(cfg) + "/" + get_lig_file(row)
            rec_file = prepare_rec(cfg, get_rec_file(row))
            out_folder = f"outputs/{program}_benchmarks/"
            os.makedirs(out_folder, exist_ok=True)
            cur_time = time.time()
            run_vina(cfg, program, out_folder, i, row, lig_file, rec_file)
            runtime = time.time() - cur_time
            runtimes.append(runtime)
        except KeyboardInterrupt:
            raise
        except:
            # raise
            print_exc()

    out_file = f"outputs/{program}_timer.txt"
    print(f"printing results to {out_file}")
    with open(out_file, "w") as f:
        f.write(f"{program} Average runtime: {sum(runtimes)/len(runtimes)} s\n")

if __name__ == "__main__":
    cfg = get_config("vina_ff")
    benchmark(cfg, "vina")
    benchmark(cfg, "gnina")