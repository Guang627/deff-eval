"""Can a small pilot predict how much concentrating a benchmark on few sources inflates the spread of pass@k?

Rule tested: SD(tier) / SD(baseline) ~ sqrt(1 + (m - 1) rho), m = problems per source, rho = intraclass correlation
of the per-problem estimate (Chen, n attempts) between sources. rho is always estimated on pilot data that are
disjoint from the data used to measure the spread (no resampling result is used twice).

GSM-Symbolic (sources = templates). Each split puts 15/15/10 templates per level (main/P1/P2) into a pilot that sees
4 variants x n = 16 attempts each (~160 problems); the remaining templates are the holdout pool. Predictions:
  naive   sqrt(1 + (V - 1) rho), rho pooled within level from the pilot;
  design  stratified two-stage variance with finite-population corrections for the actual draws
          (templates out of T_l, variants out of 20, n of 64 attempts), components from the pilot (pass@1 only).
Measured: SD of the 100-problem estimate over draws of the four tiers from the holdout pool.

SWE-bench Verified (sources = repositories, pass@1, one attempt per task). Each split takes 10 random submissions as
the pilot (rho = mean of their repository ICCs) and evaluates the other submissions. Tiers vs natural (40 tasks at
random from 500). Predictions use the actual tier sizes: naive sqrt(1 + (V - 1) rho); design adds the
finite-population corrections (R of the eligible repositories, V of each repository's tasks) and takes the
between-repository variance of the eligible repositories and their within variances from the pilot submissions.

MATH-500 (no source labels; groups = subject x level cells, which are composition variables). Design A redraws 12 of
35 cells; design B fixes the cell mix. Shows that the same formula "explains" design A even though nothing here is a
source: the cause is the composition change, and fixing the composition removes it.

Coverage is checked separately: iid normal interval vs the same interval widened by the pilot's sqrt(deff).
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from passk_smooth.estimators.base import passk_weights
from passk_smooth.experiments.common import subsample_counts, true_passk

Z = 1.959964


def anova_balanced(y: np.ndarray) -> tuple[float, float]:
    """y (groups, per-group). Returns (sigma_b^2, MSW)."""
    v = y.shape[1]
    msb = v * y.mean(1).var(ddof=1)
    msw = y.var(1, ddof=1).mean()
    return max(0.0, (msb - msw) / v), msw


def anova_unbalanced(y: np.ndarray, g: np.ndarray) -> tuple[float, float]:
    labs, gi = np.unique(g, return_inverse=True)
    k, n = labs.size, y.size
    sizes = np.bincount(gi)
    means = np.bincount(gi, y) / sizes
    msb = (sizes * (means - y.mean()) ** 2).sum() / (k - 1)
    msw = ((y - means[gi]) ** 2).sum() / (n - k)
    n0 = (n - (sizes**2).sum() / n) / (k - 1)
    return max(0.0, (msb - msw) / n0), msw


def cover(est, se, target):
    return float(np.mean(np.abs(est - target) <= Z * se))


# ---------------------------------------------------------------- GSM-Symbolic
GSM_TIERS = {"100x1": ((40, 30, 30), 1), "50x2": ((20, 15, 15), 2), "20x5": ((8, 6, 6), 5), "10x10": ((4, 3, 3), 10)}
LEVELS, W = ["main", "p1", "p2"], np.array([0.4, 0.3, 0.3])


def gsm(path: str, splits: int, reps: int, seed: int, n: int = 16, ks=(1, 16), split_by: str = "original",
        pilot_q: tuple[int, int] = (10, 5)) -> pd.DataFrame:
    """split_by="original" keeps every difficulty version (main/P1/P2) of one GSM8K question on the same side of the
    pilot/held-out split; "level" (the first version of this analysis) splits templates within each level separately,
    so the pilot and the held-out pool share ~34 of the pilot's ~35 original questions through other levels."""
    d = pd.read_csv(path)
    big_n = int(d.n_samples.iloc[0])
    wk = {k: passk_weights(n, k) for k in ks}
    tm, srcs = [], []  # per level: array (templates, 20 variants) of counts; original GSM8K id per template
    for lev in LEVELS:
        dl = d[d.level == lev].sort_values("problem_id")
        tm.append(np.stack([g.n_correct.to_numpy() for _, g in dl.groupby("subject")]))
        srcs.append(np.array([g.source.iloc[0] for _, g in dl.groupby("subject")]).astype(str))
    tri = np.unique(srcs[2])  # original questions with all three levels (50); the other 50 have main and P1 only
    duo = np.setdiff1d(np.unique(srcs[0]), tri)
    pilot_t, pilot_v = (sum(pilot_q), sum(pilot_q), pilot_q[0]), 4  # default 15/15/10 templates
    rows = []
    for s in range(splits):
        rng = np.random.default_rng([seed, 1, s])
        hold, comp = [], {k: [] for k in ks}
        if split_by == "original":  # 10 three-level + 5 two-level questions -> 15/15/10 pilot templates
            pil_src = np.concatenate([rng.choice(tri, pilot_q[0], replace=False), rng.choice(duo, pilot_q[1], replace=False)])
        pil_ids, hold_ids = set(), set()
        for li in range(3):
            perm = rng.permutation(tm[li].shape[0])
            if split_by == "original":
                inp = np.isin(srcs[li][perm], pil_src)
                perm = np.concatenate([perm[inp], perm[~inp]])
                assert inp.sum() == pilot_t[li]
            pil_ids |= set(srcs[li][perm[: pilot_t[li]]])
            hold_ids |= set(srcs[li][perm[pilot_t[li] :]])
            pc = tm[li][perm[: pilot_t[li]]][:, rng.permutation(20)[:pilot_v]]
            hold.append(tm[li][perm[pilot_t[li] :]])
            x = subsample_counts(rng, pc.ravel(), big_n, n).reshape(pc.shape)
            s2a = np.mean((1 - n / big_n) * x * (n - x) / (n * (n - 1)) / n)  # attempt noise of y_1
            for k in ks:
                sb, msw = anova_balanced(wk[k][x])
                comp[k].append((sb, msw, s2a))
        shared = len(pil_ids & hold_ids)
        rho = {k: float(sum(W[li] * comp[k][li][0] for li in range(3))
                        / sum(W[li] * (comp[k][li][0] + comp[k][li][1]) for li in range(3))) for k in ks}  # fmt: skip
        tl = [h.shape[0] for h in hold]

        def design_var(t, v, comp=comp, tl=tl, fpc=True):
            out = 0.0
            for li in range(3):
                sb, msw, s2a = comp[1][li]
                sw = max(0.0, msw - s2a)
                f1, f2 = ((1 - t[li] / tl[li]), (1 - v / 20)) if fpc else (1.0, 1.0)
                out += W[li] ** 2 * (f1 * sb / t[li] + f2 * sw / (t[li] * v) + s2a / (t[li] * v))
            return out

        target = {k: sum(W[li] * true_passk(hold[li].ravel(), big_n, np.array([k]))[0] for li in range(3)) for k in ks}
        res = {}
        for tier, (t, v) in GSM_TIERS.items():
            est = {k: np.empty(reps) for k in ks}
            se = {k: np.empty(reps) for k in ks}
            for r in range(reps):
                cs = []
                for li in range(3):
                    tt = rng.choice(tl[li], t[li], replace=False)
                    vv = np.argsort(rng.random((t[li], 20)), axis=1)[:, :v]
                    cs.append(hold[li][tt[:, None], vv].ravel())
                x = subsample_counts(rng, np.concatenate(cs), big_n, n)
                for k in ks:
                    y = wk[k][x]
                    est[k][r], se[k][r] = y.mean(), y.std(ddof=1) / np.sqrt(y.size)
            res[tier] = (est, se, v, t)
        for tier, (est, se, v, t) in res.items():
            for k in ks:
                deff = 1 + (v - 1) * rho[k]
                rows.append({
                    "benchmark": "GSM-Symbolic", "split": s, "tier": tier, "k": k, "m": v, "rho_pilot": rho[k],
                    "pred_naive": np.sqrt(deff),
                    "pred_design": np.sqrt(design_var(t, v) / design_var(*GSM_TIERS["100x1"])) if k == 1 else np.nan,
                    "pred_design_nofpc": np.sqrt(design_var(t, v, fpc=False) / design_var(*GSM_TIERS["100x1"], fpc=False)) if k == 1 else np.nan,
                    "measured": est[k].std() / res["100x1"][0][k].std(), "sd": est[k].std(),
                    "sd_design": np.sqrt(design_var(t, v)) if k == 1 else np.nan,
                    "cov_iid": cover(est[k], se[k], target[k]),
                    "cov_deff": cover(est[k], se[k] * np.sqrt(deff), target[k]), "shared_original_ids": shared,
                })  # fmt: skip
        print("gsm split", s, {k: round(v, 3) for k, v in rho.items()}, flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- SWE-bench
SWE_TIERS = {"natural": (None, None), "8x5": (8, 5), "4x10": (4, 10), "2x20": (2, 20), "1x40": (1, 40)}


def swe(path: str, splits: int, reps: int, seed: int, pilot: int = 10, m: int = 40) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = pd.read_csv(path)
    Y = d.pivot(index="submission", columns="instance_id", values="resolved")
    r = Y.mean(axis=1)
    Y = Y.loc[(r >= 0.1) & (r <= 0.9)]
    repo = np.array([i.rsplit("-", 1)[0] for i in Y.columns])
    labs, ri = np.unique(repo, return_inverse=True)
    members = [np.flatnonzero(ri == j) for j in range(labs.size)]
    sizes = np.array([mm.size for mm in members])
    A = Y.to_numpy(float)
    icc = np.array([(lambda sb, sw: sb / (sb + sw))(*anova_unbalanced(a, ri)) for a in A])
    # repository ICC after removing difficulty (terciles of the other submissions' solve rate)
    icc_resid = []
    for i, a in enumerate(A):
        other = np.delete(A, i, 0).mean(0)
        st = np.digitize(other, np.quantile(other, [1 / 3, 2 / 3]))
        res = a - (np.bincount(st, a) / np.bincount(st))[st]
        sb, sw = anova_unbalanced(res, ri)
        icc_resid.append(sb / (sb + sw))
    subs = pd.DataFrame({"submission": Y.index, "rate": A.mean(1), "icc_repo": icc, "icc_repo_given_difficulty": icc_resid})

    def design_ratio(pil, R, V, fpc=True):
        """Exact-design variance ratio with repository means and within variances taken from pilot submissions."""
        if R is None:
            return 1.0
        elig = np.flatnonzero(sizes >= V)
        out = []
        for a in A[pil]:
            mu = np.array([a[members[j]].mean() for j in elig])
            s2 = np.array([a[members[j]].var(ddof=1) for j in elig])
            if fpc:
                var_t = (1 - R / elig.size) * mu.var(ddof=1) / R + np.mean((1 - V / sizes[elig]) * s2 / V) / R
                var_n = (1 - m / a.size) * a.var(ddof=1) / m
            else:
                var_t = mu.var(ddof=1) / R + np.mean(s2 / V) / R
                var_n = a.var(ddof=1) / m
            out.append(var_t / var_n)
        return float(np.sqrt(np.mean(out)))

    # measured SD / coverage per submission (computed once; pilot splits only change rho)
    meas = []
    for si in range(A.shape[0]):
        rng = np.random.default_rng([seed, 2, si])
        y = A[si]
        for tier, (R, V) in SWE_TIERS.items():
            est, se = np.empty(reps), np.empty(reps)
            for rep in range(reps):
                if R is None:
                    idx = rng.choice(y.size, m, replace=False)
                else:
                    pick = rng.choice(np.flatnonzero(sizes >= V), R, replace=False)
                    idx = np.concatenate([rng.choice(members[j], V, replace=False) for j in pick])
                yy = y[idx]
                est[rep], se[rep] = yy.mean(), max(yy.std(ddof=1), 1e-9) / np.sqrt(m)
            meas.append({"si": si, "tier": tier, "sd": est.std(), "est": est, "se": se, "target": y.mean()})
    meas_df = pd.DataFrame(meas)
    base = meas_df[meas_df.tier == "natural"].set_index("si").sd
    rows = []
    for s in range(splits):
        rng = np.random.default_rng([seed, 3, s])
        pil = rng.choice(A.shape[0], pilot, replace=False)
        rho = float(icc[pil].mean())
        held = np.setdiff1d(np.arange(A.shape[0]), pil)
        for tier, (R, V) in SWE_TIERS.items():
            mt = meas_df[(meas_df.tier == tier) & meas_df.si.isin(held)]
            deff = 1.0 if R is None else 1 + (V - 1) * rho
            ratios = mt.sd.to_numpy() / base.loc[mt.si].to_numpy()
            rows.append({
                "benchmark": "SWE-bench Verified", "split": s, "tier": tier, "k": 1, "m": V or 1, "rho_pilot": rho,
                "pred_naive": np.sqrt(deff), "pred_design": design_ratio(pil, R, V), "pred_design_nofpc": design_ratio(pil, R, V, fpc=False),
                "measured": float(np.median(ratios)), "measured_q25": float(np.quantile(ratios, 0.25)),
                "measured_q75": float(np.quantile(ratios, 0.75)), "sd": float(mt.sd.mean()),
                "cov_iid": float(np.mean([cover(q.est, q.se, q.target) for q in mt.itertuples()])),
                "cov_deff": float(np.mean([cover(q.est, q.se * np.sqrt(deff), q.target) for q in mt.itertuples()])),
            })  # fmt: skip
    return pd.DataFrame(rows), subs


# ---------------------------------------------------------------- MATH-500
def math(path: str, splits: int, reps: int, seed: int, n: int = 16, k: int = 1) -> tuple[pd.DataFrame, dict]:
    d = pd.read_csv(path)
    c, big_n = d.n_correct.to_numpy(), int(d.n_samples.iloc[0])
    cell = d.cluster.to_numpy()
    lev = d.level.to_numpy()
    w = passk_weights(n, k)
    rows, rhos = [], []
    for s in range(splits):
        rng = np.random.default_rng([seed, 4, s])
        perm = rng.permutation(c.size)
        pil, hold = perm[:150], perm[150:]
        y = w[subsample_counts(rng, c[pil], big_n, n)]
        sb, sw = anova_unbalanced(y, cell[pil])
        rho_cell = sb / (sb + sw)
        sb, sw = anova_unbalanced(y, lev[pil])
        rho_level = sb / (sb + sw)
        res = y - pd.Series(y).groupby(lev[pil]).transform("mean").to_numpy()
        sb, sw = anova_unbalanced(res, cell[pil])
        rho_cell_given_level = sb / (sb + sw)
        rhos.append({"rho_cell": rho_cell, "rho_level": rho_level, "rho_subject_within_level": rho_cell_given_level})
        hc = cell[hold]
        labs, gi = np.unique(hc, return_inverse=True)
        mem = [hold[gi == j] for j in range(labs.size)]
        gs = np.array([mm.size for mm in mem])
        mbar = (gs**2).sum() / gs.sum()
        m_b = 12 / labs.size * hold.size
        alloc = np.maximum(1, np.round(m_b * gs / hold.size)).astype(int)
        target = true_passk(c[hold], big_n, np.array([k]))[0]
        out = {}
        for design in ["A_group_redraw", "B_fixed_mix"]:
            est, se = np.empty(reps), np.empty(reps)
            for r in range(reps):
                if design == "A_group_redraw":
                    idx = np.concatenate([mem[j] for j in rng.choice(labs.size, 12, replace=False)])
                else:
                    idx = np.concatenate([rng.choice(mem[j], alloc[j], replace=True) for j in range(labs.size)])
                yy = w[subsample_counts(rng, c[idx], big_n, n)]
                est[r], se[r] = yy.mean(), yy.std(ddof=1) / np.sqrt(yy.size)
            out[design] = (est, se)
        pred_a = 1 + (mbar - 1) * rho_cell
        pred_b = 1 - rho_cell
        for design, deff in [("A_group_redraw", pred_a), ("B_fixed_mix", pred_b)]:
            est, se = out[design]
            rows.append({
                "benchmark": "MATH-500", "split": s, "tier": design, "k": k, "m": mbar if design[0] == "A" else 1,
                "rho_pilot": rho_cell, "pred_naive": np.sqrt(pred_a / pred_b) if design[0] == "A" else 1.0,
                "pred_design": np.nan, "measured": est.std() / out["B_fixed_mix"][0].std(), "sd": est.std(),
                "cov_iid": cover(est, se, target), "cov_deff": cover(est, se * np.sqrt(max(deff, 1e-9)), target),
            })  # fmt: skip
        print("math split", s, round(rho_cell, 3), flush=True)
    rh = pd.DataFrame(rhos)
    return pd.DataFrame(rows), {c_: [float(rh[c_].median()), float(rh[c_].quantile(0.05)), float(rh[c_].quantile(0.95))]
                                for c_ in rh}  # fmt: skip


def plot(df: pd.DataFrame, out: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6), sharex=False)
    series = [
        ("GSM-Symbolic", 1, "pred_naive", "C0", "o", "GSM pass@1, naive"),
        ("GSM-Symbolic", 1, "pred_design", "C0", "s", "GSM pass@1, design"),
        ("GSM-Symbolic", 16, "pred_naive", "C2", "^", "GSM pass@16, naive"),
        ("SWE-bench Verified", 1, "pred_naive", "C1", "o", "SWE-bench, naive"),
        ("SWE-bench Verified", 1, "pred_design", "C1", "s", "SWE-bench, design"),
        ("MATH-500", 1, "pred_naive", "C3", "D", "MATH-500 cells (composition)"),
    ]
    for b, k, col, colour, mk, lab in series:
        g = df[(df.benchmark == b) & (df.k == k)].dropna(subset=[col]).groupby("tier", sort=False)
        px, py = g[col].median(), g.measured.median()
        xe = np.vstack([px - g[col].quantile(0.05), g[col].quantile(0.95) - px])
        if b == "SWE-bench Verified":  # spread across held-out submissions (IQR), not across pilot splits
            ye = np.vstack([py - g.measured_q25.median(), g.measured_q75.median() - py])
        else:
            ye = np.vstack([py - g.measured.quantile(0.05), g.measured.quantile(0.95) - py])
        a = ax[0] if b != "SWE-bench Verified" else ax[1]
        a.errorbar(px, py, xerr=xe, yerr=ye, fmt=mk, color=colour, mfc="white" if col == "pred_naive" else colour,
                   capsize=2, lw=1, label=lab)  # fmt: skip
    for a, lim in zip(ax, [(0.9, 4.2), (0.85, 1.65)], strict=True):
        a.plot(lim, lim, color="grey", lw=1, ls="--")
        a.set(xlim=lim, ylim=lim, xlabel="predicted SD ratio (pilot rho)", ylabel="measured SD ratio (held-out)")
        a.spines[["top", "right"]].set_visible(False)
        a.legend(frameon=False, fontsize=8, loc="upper left")
    ax[0].set_title("GSM-Symbolic tiers vs 100x1; MATH-500 design A vs B")
    ax[1].set_title("SWE-bench tiers vs natural (median over submissions)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_deff_pred_vs_measured.png"), dpi=150)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gsm", default="results/gsm_symbolic_qwen2.5-math-1.5b/counts.csv")
    ap.add_argument("--swe", default="results/swebench_verified/resolved_matrix.csv.gz")
    ap.add_argument("--math", default="results/math500_qwen2.5-math-1.5b/counts.csv")
    ap.add_argument("--out", default="results/deff_validation")
    ap.add_argument("--splits", type=int, default=20)
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gsm-split", choices=["original", "level"], default="original")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    g = gsm(a.gsm, a.splits, a.reps, a.seed, split_by=a.gsm_split)
    sw, subs = swe(a.swe, 100, 300, a.seed)
    mt, mrho = math(a.math, a.splits, a.reps, a.seed)
    df = pd.concat([g, sw, mt], ignore_index=True)
    df.to_csv(os.path.join(a.out, "deff_splits.csv"), index=False)
    subs.to_csv(os.path.join(a.out, "swe_submission_icc.csv"), index=False)
    agg = (
        df.groupby(["benchmark", "k", "tier"], sort=False)
        .agg(
            m=("m", "median"),
            rho_pilot=("rho_pilot", "median"),
            rho_p05=("rho_pilot", lambda x: x.quantile(0.05)),
            rho_p95=("rho_pilot", lambda x: x.quantile(0.95)),
            pred_naive=("pred_naive", "median"),
            pred_design=("pred_design", "median"),
            measured=("measured", "median"),
            measured_p05=("measured", lambda x: x.quantile(0.05)),
            measured_p95=("measured", lambda x: x.quantile(0.95)),
            cov_iid=("cov_iid", "mean"),
            cov_deff=("cov_deff", "mean"),
        )  # fmt: skip
        .reset_index()
    )
    agg.to_csv(os.path.join(a.out, "deff_summary.csv"), index=False)
    q = subs[["icc_repo", "icc_repo_given_difficulty"]].quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    meta = {"swe_icc_quantiles": q.round(4).to_dict(),
            "math_rho_median_p05_p95": mrho, "splits": a.splits, "reps": a.reps}  # fmt: skip
    with open(os.path.join(a.out, "deff_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    with open(os.path.join(a.out, "table_deff.md"), "w") as fh:
        fh.write(agg.round(3).to_markdown(index=False) + "\n")
    plot(df, a.out)
    print(agg.round(3).to_string(index=False))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
