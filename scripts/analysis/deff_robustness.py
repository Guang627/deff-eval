import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import deff_validation as dv  # noqa: E402

OUT = "results/deff_validation"
GSM = "results/gsm_symbolic_qwen2.5-math-1.5b/counts.csv"
SWE = "results/swebench_verified/resolved_matrix.csv.gz"


def summarise(df, label):
    df = df[df.tier.isin(["50x2", "20x5", "10x10", "8x5", "4x10", "2x20", "1x40"])].copy()
    df["err_design"] = df.pred_design / df.measured - 1
    df["err_naive"] = df.pred_naive / df.measured - 1
    df["err_nofpc"] = df.pred_design_nofpc / df.measured - 1
    g = df.groupby(["benchmark", "k", "tier"], sort=False)
    out = g.agg(
        rho=("rho_pilot", "median"), rho_p05=("rho_pilot", lambda x: x.quantile(0.05)),
        rho_p95=("rho_pilot", lambda x: x.quantile(0.95)), measured=("measured", "median"),
        pred_design=("pred_design", "median"), pred_design_nofpc=("pred_design_nofpc", "median"),
        pred_naive=("pred_naive", "median"),
        err_design_med=("err_design", "median"), err_design_p05=("err_design", lambda x: x.quantile(0.05)),
        err_design_p95=("err_design", lambda x: x.quantile(0.95)),
        abs_err_design_med=("err_design", lambda x: x.abs().median()),
        err_naive_med=("err_naive", "median"), err_nofpc_med=("err_nofpc", "median"),
        cov_deff=("cov_deff", "mean"),
    ).reset_index()  # fmt: skip
    out.insert(0, "pilot", label)
    return out


def main():
    rows = []
    for q in [(4, 2), (6, 3), (10, 5), (16, 8)]:
        g = dv.gsm(GSM, 20, 1000, 0, pilot_q=q)
        rows.append(summarise(g, f"GSM {sum(q)} questions"))
        print("gsm", q, flush=True)
    for p in [3, 5, 10, 20]:
        s, _ = dv.swe(SWE, 100, 300, 0, pilot=p)
        rows.append(summarise(s, f"SWE {p} submissions"))
        print("swe", p, flush=True)
    r = pd.concat(rows, ignore_index=True)
    r.to_csv(os.path.join(OUT, "robustness_pilot_size.csv"), index=False)
    print(r.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
