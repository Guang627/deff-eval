import os
import sys

import pandas as pd

COLS = ["benchmark", "split", "tier", "k", "m", "rho_pilot", "pred_naive", "pred_design", "measured", "sd", "sd_design",
        "cov_iid", "cov_deff", "shared_original_ids"]  # fmt: skip

if __name__ == "__main__":
    src, split = sys.argv[1], sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else "results/deff_validation"
    d = pd.read_csv(src)
    d = d[d.benchmark == "GSM-Symbolic"][COLS]
    n = d.split.nunique()
    path = os.path.join(out, f"gsm_{split}_{n}.csv")
    d.to_csv(path, index=False)
    print(path, len(d))
