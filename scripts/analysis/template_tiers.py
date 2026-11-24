"""Same difficulty mix, different numbers of templates: how reliable is pass@k for a 100-problem benchmark?

Input: counts.csv from `passk generate` on GSM-Symbolic (configs/generate_gsm_symbolic.yaml), one row per
variant with columns level (main / p1 / p2), subject (template id), n_samples (N), n_correct.
A template is (level, subject).

Each draw builds a 100-problem benchmark with a fixed composition of 40 main / 30 P1 / 30 P2 problems and
one of four tiers, templates x variants per template:
    100x1 : 40/30/30 templates, 1 variant each
    50x2  : 20/15/15 templates, 2 variants each
    20x5  :  8/6/6   templates, 5 variants each
    10x10 :  4/3/3   templates, 10 variants each
Templates are drawn without replacement within level, variants without replacement within template, and n of
the N attempts per problem without replacement. Every tier uses the same number of problems and attempts.

Target (fixed before any draw): 0.4 pass@k(main) + 0.3 pass@k(P1) + 0.3 pass@k(P2), each level's pass@k being
the mean over all its templates and variants from all N attempts.
Estimate: Chen's estimator averaged over the 100 problems (the composition already matches the weights).
Intervals (95% percentile bootstrap): iid over problems; stratified by level; template-cluster bootstrap
stratified by level. Reported per tier and k: SD and RMSE of the estimate, coverage and mean width.

--synthetic builds a pool with logit p = level shift + template effect + variant effect, for dry runs.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from passk_smooth.estimators.base import passk_weights
from passk_smooth.experiments.common import subsample_counts, true_passk

LEVELS = ["main", "p1", "p2"]
WEIGHTS = np.array([0.4, 0.3, 0.3])
TIERS = {"100x1": ((40, 30, 30), 1), "50x2": ((20, 15, 15), 2), "20x5": ((8, 6, 6), 5), "10x10": ((4, 3, 3), 10)}


def synthetic_pool(seed: int, big_n: int, variants: int, sd_t: float, sd_v: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for lev, n_t, shift in [("main", 100, 1.5), ("p1", 100, 0.3), ("p2", 50, -0.8)]:
        for t in range(n_t):
            u = shift + rng.normal(0, sd_t)
            for v in range(variants):
                p = 1 / (1 + np.exp(-(u + rng.normal(0, sd_v))))
                rows.append({"level": lev, "subject": f"t{t:03d}", "instance": v, "n_samples": big_n,
                             "n_correct": rng.binomial(big_n, p)})  # fmt: skip
    return pd.DataFrame(rows)


def boot_ci(y: np.ndarray, groups: list[np.ndarray], rng: np.random.Generator, n_boot: int) -> np.ndarray:
    """Percentile CI of the mean of y (m, K). groups: per stratum, a list of index arrays (resampling units)."""
    m = y.shape[0]
    stats = np.zeros((n_boot, y.shape[1]))
    for units in groups:
        sums = np.array([y[u].sum(0) for u in units])  # (units, K)
        pick = rng.integers(0, len(units), size=(n_boot, len(units)))
        stats += sums[pick].sum(1)
    return np.quantile(stats / m, [0.025, 0.975], axis=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--counts", help="GSM-Symbolic counts.csv")
    ap.add_argument("--synthetic", action="store_true", help="use a simulated pool instead of --counts")
    ap.add_argument("--sd-template", type=float, default=1.2)
    ap.add_argument("--sd-variant", type=float, default=0.8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--ks", default="1,4,8,16")
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d = synthetic_pool(a.seed, 64, 20, a.sd_template, a.sd_variant) if a.synthetic else pd.read_csv(a.counts)
    d["level"] = d["level"].astype(str).str.lower()
    big_n = int(d.n_samples.iloc[0])
    ks = np.array([int(k) for k in a.ks.split(",")])
    w = np.array([passk_weights(a.n, int(k)) for k in ks])  # (K, n+1)
    templates = []  # per level: list of index arrays (variants of one template)
    target = np.zeros(ks.size)
    for li, lev in enumerate(LEVELS):
        dl = d[d.level == lev]
        if dl.empty:
            raise SystemExit(f"level {lev} missing from counts")
        target += WEIGHTS[li] * true_passk(dl.n_correct.to_numpy(), big_n, ks)
        templates.append([g.index.to_numpy() for _, g in dl.groupby("subject")])
    c = d.n_correct.to_numpy()
    icc = {}
    for lev in LEVELS:  # one-way ANOVA share of between-template variance in per-variant success rates
        dl = d[d.level == lev].assign(ph=lambda z: z.n_correct / big_n)
        g = dl.groupby("subject").ph
        v_n = int(g.size().iloc[0])
        msb, msw = v_n * g.mean().var(), g.var().mean()
        s2b = max(0.0, (msb - msw) / v_n)
        icc[lev] = round(float(s2b / (s2b + msw)), 3)
    print("between-template share of variance (pass@1 scale):", icc)
    min_v = min(len(ix) for lev in templates for ix in lev)
    rows = []
    for tier, (n_t, v) in TIERS.items():
        if v > min_v:
            raise SystemExit(f"tier {tier} needs {v} variants per template; pool has {min_v}")
        for rep in range(a.reps):
            rng = np.random.default_rng([a.seed, rep, v])
            idx, lev_of, tmpl_of = [], [], []
            for li in range(3):
                for j, t in enumerate(rng.choice(len(templates[li]), n_t[li], replace=False)):
                    vs = rng.choice(templates[li][t], v, replace=False)
                    idx += list(vs)
                    lev_of += [li] * v
                    tmpl_of += [j] * v
            idx_a, lev_a, tmpl_a = np.array(idx), np.array(lev_of), np.array(tmpl_of)
            x = subsample_counts(rng, c[idx_a], big_n, a.n)
            y = w[:, x].T  # (100, K) per-problem Chen
            est = y.mean(0)
            schemes = {
                "iid": [[np.array([i]) for i in range(y.shape[0])]],
                "stratified": [[np.array([i]) for i in np.flatnonzero(lev_a == li)] for li in range(3)],
                "template_cluster": [
                    [np.flatnonzero((lev_a == li) & (tmpl_a == j)) for j in range(n_t[li])] for li in range(3)
                ],
            }
            for s, groups in schemes.items():
                lo, hi = boot_ci(y, groups, rng, a.boot)
                for kk in range(ks.size):
                    rows.append({"tier": tier, "rep": rep, "scheme": s, "k": int(ks[kk]), "target": target[kk],
                                 "estimate": est[kk], "lower": lo[kk], "upper": hi[kk]})  # fmt: skip
    raw = pd.DataFrame(rows)
    raw["covered"] = (raw.lower <= raw.target) & (raw.target <= raw.upper)
    raw["width"] = raw.upper - raw.lower
    raw["err"] = raw.estimate - raw.target
    summ = (
        raw.groupby(["tier", "k", "scheme"], sort=False)
        .agg(target=("target", "first"), bias=("err", "mean"), sd=("estimate", "std"),
             rmse=("err", lambda e: float(np.sqrt(np.mean(e**2)))), coverage=("covered", "mean"),
             mean_width=("width", "mean"))  # fmt: skip
        .reset_index()
    )
    raw.to_csv(os.path.join(a.out, "template_tiers_raw.csv.gz"), index=False)
    summ.to_csv(os.path.join(a.out, "template_tiers_summary.csv"), index=False)
    pp = summ.copy()
    for col in ["target", "bias", "sd", "rmse", "mean_width"]:
        pp[col] = (100 * pp[col]).round(2)
    pp["coverage"] = pp["coverage"].round(3)
    head = (f"{'synthetic pool' if a.synthetic else a.counts}; N = {big_n}, n = {a.n}, {a.reps} draws, "
            f"{a.boot} bootstrap resamples; values in pp except coverage.\n\n")  # fmt: skip
    with open(os.path.join(a.out, "table_template_tiers.md"), "w") as fh:
        fh.write(head + pp.to_markdown(index=False) + "\n")
    meta = {"N": big_n, "n": a.n, "ks": ks.tolist(), "reps": a.reps, "boot": a.boot, "synthetic": a.synthetic,
            "tiers": {k: {"templates": list(t), "variants": v} for k, (t, v) in TIERS.items()},
            "icc_template": icc, "weights": dict(zip(LEVELS, WEIGHTS.tolist(), strict=True))}  # fmt: skip
    with open(os.path.join(a.out, "template_tiers_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print(pp[pp.k == ks[0]].to_string(index=False))


if __name__ == "__main__":
    main()
