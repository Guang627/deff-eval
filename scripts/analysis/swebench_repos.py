"""SWE-bench Verified: same task budget, fewer or more repositories. pass@1 only, CPU only.

Data: per-task resolved lists of public submissions (github.com/SWE-bench/experiments,
evaluation/verified/*/results/results.json). Each submission is one model+agent run, one attempt per task.
`--build` turns a checkout into resolved_matrix.csv.gz (submission x instance), saved in the repo.

For every submission separately, each draw builds a 40-task benchmark:
    natural : 40 tasks at random from all 500 (repositories in their natural proportions)
    8x5, 4x10, 2x20, 1x40 : R repositories x V tasks each; repositories drawn uniformly among those with at least V
        tasks, tasks without replacement within repository.
Target (fixed in advance): the submission's resolve rate over all 500 tasks (task-equal). Sensitivity target:
mean over the 12 repositories of the per-repository resolve rate (repository-equal).
Difficulty: 3 strata by the share of the *other* submissions that solved the task (terciles); the estimate
"poststrat" reweights the drawn tasks to the pool's difficulty mix, "plain" is the raw mean.
Intervals (95%): iid bootstrap, difficulty-stratified bootstrap (poststrat estimate), repository-cluster bootstrap,
Wilson. Repository share of variance (one-way ANOVA on 0/1 outcomes) is reported per submission.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from passk_smooth.inference import wilson_interval

TIERS = {"natural": (None, None), "8x5": (8, 5), "4x10": (4, 10), "2x20": (2, 20), "1x40": (1, 40)}
M = 40


def build(src: str) -> pd.DataFrame:
    sub, ids = {}, set()
    for f in sorted(glob.glob(os.path.join(src, "evaluation/verified/*/results/results.json"))):
        with open(f) as fh:
            d = json.load(fh)
        sub[f.split(os.sep)[-3]] = set(d.get("resolved", []))
        ids |= set(d.get("resolved", [])) | set(d.get("generated", [])) | set(d.get("no_generation", []))
    ids_s = sorted(ids)
    rows = [(s, i, int(i in r)) for s, r in sub.items() for i in ids_s]
    return pd.DataFrame(rows, columns=["submission", "instance_id", "resolved"])


def boot_mean(y: np.ndarray, units: list[list[np.ndarray]], wts: np.ndarray | None, rng, n_boot: int) -> np.ndarray:
    """Percentile CI. units: per stratum, list of index arrays. wts: stratum weights (None = pooled mean)."""
    out = []
    for us in units:
        sums = np.array([y[u].sum() for u in us])
        cnts = np.array([u.size for u in us])
        pick = rng.integers(0, len(us), size=(n_boot, len(us)))
        out.append((sums[pick].sum(1), cnts[pick].sum(1)))
    if wts is None:
        stat = sum(s for s, _ in out) / sum(c for _, c in out)
    else:
        stat = sum(w * s / np.maximum(c, 1) for w, (s, c) in zip(wts, out, strict=True))
    return np.quantile(stat, [0.025, 0.975])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", help="path to a SWE-bench/experiments checkout; writes the matrix and exits")
    ap.add_argument("--out", default="results/swebench_verified")
    ap.add_argument("--matrix", default="results/swebench_verified/resolved_matrix.csv.gz")
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--min-rate", type=float, default=0.1)
    ap.add_argument("--max-rate", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    mpath = a.matrix
    if a.build:
        build(a.build).to_csv(mpath, index=False)
        print("wrote", mpath)
        return
    d = pd.read_csv(mpath)
    Y = d.pivot(index="submission", columns="instance_id", values="resolved")
    repo_lab = np.array([i.rsplit("-", 1)[0] for i in Y.columns])
    repos, ridx = np.unique(repo_lab, return_inverse=True)
    members = [np.flatnonzero(ridx == r) for r in range(repos.size)]
    sizes = np.array([m.size for m in members])
    rates = Y.mean(axis=1)
    keep = rates[(rates >= a.min_rate) & (rates <= a.max_rate)].index
    Yall = Y.to_numpy(float)
    rows, icc_rows = [], []
    for si, s in enumerate(keep):
        rng = np.random.default_rng([a.seed, si])
        y = Y.loc[s].to_numpy(float)
        others = np.delete(Yall, Y.index.get_loc(s), axis=0).mean(0)
        strat = np.digitize(others, np.quantile(others, [1 / 3, 2 / 3]))  # 0 hard .. 2 easy
        wts = np.bincount(strat, minlength=3) / y.size
        t_task = y.mean()
        t_repo = np.mean([y[m].mean() for m in members])
        # repository share of variance (ANOVA, unbalanced: n0 correction)
        k, n_tot = repos.size, y.size
        msb = sum(m.size * (y[m].mean() - t_task) ** 2 for m in members) / (k - 1)
        msw = sum(((y[m] - y[m].mean()) ** 2).sum() for m in members) / (n_tot - k)
        n0 = (n_tot - (sizes**2).sum() / n_tot) / (k - 1)
        s2b = max(0.0, (msb - msw) / n0)
        icc_rows.append({"submission": s, "rate": t_task, "repo_equal": t_repo, "icc_repo": s2b / (s2b + msw)})
        for tier, (R, V) in TIERS.items():
            for rep in range(a.reps):
                if R is None:
                    idx = rng.choice(y.size, M, replace=False)
                else:
                    elig = np.flatnonzero(sizes >= V)
                    pick = rng.choice(elig, R, replace=False)
                    idx = np.concatenate([rng.choice(members[r], V, replace=False) for r in pick])
                yy, st, rr = y[idx], strat[idx], ridx[idx]
                plain = yy.mean()
                present = [g for g in range(3) if (st == g).any()]
                w_p = wts[present] / wts[present].sum()
                post = sum(w * yy[st == g].mean() for w, g in zip(w_p, present, strict=True))
                ivs = {
                    "iid": (plain, boot_mean(yy, [[np.array([i]) for i in range(M)]], None, rng, a.boot)),
                    "stratified": (post, boot_mean(yy, [[np.array([i]) for i in np.flatnonzero(st == g)] for g in present],
                                                   w_p, rng, a.boot)),  # fmt: skip
                    "repo_cluster": (plain, boot_mean(yy, [[np.flatnonzero(rr == r) for r in np.unique(rr)]], None, rng, a.boot)),
                    "wilson": (plain, np.array(wilson_interval(int(yy.sum()), M))),
                }
                for sc, (est, (lo, hi)) in ivs.items():
                    rows.append({"submission": s, "tier": tier, "rep": rep, "scheme": sc, "estimate": est,
                                 "lower": lo, "upper": hi, "t_task": t_task, "t_repo": t_repo})  # fmt: skip
    raw = pd.DataFrame(rows)
    for t in ["t_task", "t_repo"]:
        raw[f"cov_{t}"] = (raw.lower <= raw[t]) & (raw[t] <= raw.upper)
        raw[f"err_{t}"] = raw.estimate - raw[t]
    raw["width"] = raw.upper - raw.lower
    per_sub = (
        raw.groupby(["tier", "scheme", "submission"], sort=False)
        .agg(sd=("estimate", "std"), bias=("err_t_task", "mean"), rmse=("err_t_task", lambda e: np.sqrt(np.mean(e**2))),
             cov=("cov_t_task", "mean"), cov_repo_target=("cov_t_repo", "mean"), width=("width", "mean"))  # fmt: skip
        .reset_index()
    )
    summ = per_sub.drop(columns="submission").groupby(["tier", "scheme"], sort=False).agg(["mean", "median"]).reset_index()
    summ.columns = ["_".join(c).strip("_") for c in summ.columns]
    per_sub.to_csv(os.path.join(a.out, "per_submission.csv"), index=False)
    summ.to_csv(os.path.join(a.out, "summary.csv"), index=False)
    icc = pd.DataFrame(icc_rows)
    icc.to_csv(os.path.join(a.out, "submissions.csv"), index=False)
    show = summ[["tier", "scheme", "sd_mean", "bias_mean", "rmse_mean", "cov_mean", "cov_repo_target_mean", "width_mean"]].copy()
    for c in ["sd_mean", "bias_mean", "rmse_mean", "width_mean"]:
        show[c] = (100 * show[c]).round(2)
    show[["cov_mean", "cov_repo_target_mean"]] = show[["cov_mean", "cov_repo_target_mean"]].round(3)
    head = (f"{len(keep)} submissions (resolve rate {a.min_rate}-{a.max_rate}), {M} tasks per benchmark, {a.reps} draws "
            f"per submission, {a.boot} bootstrap resamples. Averages over submissions; pp except coverage. "
            f"Median repository share of variance: {icc.icc_repo.median():.3f}.\n\n")  # fmt: skip
    with open(os.path.join(a.out, "table_swebench_repos.md"), "w") as fh:
        fh.write(head + show.to_markdown(index=False) + "\n")
    print(head + show.to_string(index=False))
    print(icc.describe().round(3).to_string())


if __name__ == "__main__":
    main()
