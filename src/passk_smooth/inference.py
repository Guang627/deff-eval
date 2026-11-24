from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .estimators.base import PassAtKEstimator

Scheme = Literal["iid", "cluster", "block"]


@dataclass
class Interval:
    k: np.ndarray
    estimate: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    se: np.ndarray
    scheme: str
    level: float


def _resample_indices(rng: np.random.Generator, m: int, scheme: Scheme, clusters: np.ndarray | None, block: int) -> np.ndarray:
    if scheme == "iid":
        return rng.integers(0, m, m)
    if scheme == "cluster":
        if clusters is None:
            raise ValueError("cluster bootstrap needs cluster labels")
        labels, inv = np.unique(clusters, return_inverse=True)
        members = [np.flatnonzero(inv == c) for c in range(labels.size)]
        pick = rng.integers(0, labels.size, labels.size)
        return np.concatenate([members[c] for c in pick])
    if scheme == "block":
        nb = int(np.ceil(m / block))
        starts = rng.integers(0, m - block + 1, nb) if m > block else np.zeros(nb, int)
        return np.concatenate([np.arange(s, min(s + block, m)) for s in starts])[:m]
    raise ValueError(f"unknown scheme {scheme!r}")


def bootstrap_ci(
    counts: np.ndarray,
    n: int,
    make_estimator: Callable[[], PassAtKEstimator],
    ks: Sequence[int],
    scheme: Scheme = "iid",
    clusters: np.ndarray | None = None,
    block: int = 10,
    n_boot: int = 500,
    level: float = 0.95,
    seed: int = 0,
) -> Interval:
    counts = np.asarray(counts)
    ks_arr = np.asarray(ks, int)
    est = make_estimator().fit(counts, n).pass_at_k(ks_arr)
    rng = np.random.default_rng(seed)
    reps = np.empty((n_boot, ks_arr.size))
    for b in range(n_boot):
        idx = _resample_indices(rng, counts.size, scheme, clusters, block)
        reps[b] = make_estimator().fit(counts[idx], n).pass_at_k(ks_arr)
    alpha = (1 - level) / 2
    lo, hi = np.nanquantile(reps, [alpha, 1 - alpha], axis=0)
    return Interval(ks_arr, est, lo, hi, np.nanstd(reps, axis=0, ddof=1), scheme, level)


def cluster_robust_se(contrib: np.ndarray, clusters: np.ndarray | None = None) -> float:
    c = np.asarray(contrib, float)
    m = c.size
    r = c - c.mean()
    if clusters is None:
        return float(np.sqrt(r @ r / (m * (m - 1))))
    _, inv = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inv, weights=r)
    g = sums.size
    adj = g / (g - 1) if g > 1 else 1.0
    return float(np.sqrt(adj * (sums @ sums)) / m)


def design_effect(contrib: np.ndarray, clusters: np.ndarray) -> float:
    return (cluster_robust_se(contrib, clusters) / cluster_robust_se(contrib)) ** 2


def wilson_interval(successes: int, m: int, level: float = 0.95) -> tuple[float, float]:
    from scipy.stats import norm

    if m == 0:
        return (np.nan, np.nan)
    z = float(norm.ppf(0.5 + level / 2))
    ph = successes / m
    den = 1 + z * z / m
    mid = (ph + z * z / (2 * m)) / den
    half = z * np.sqrt(ph * (1 - ph) / m + z * z / (4 * m * m)) / den
    return (float(mid - half), float(mid + half))


def paired_bootstrap_diff(
    counts_a: np.ndarray,
    counts_b: np.ndarray,
    n: int,
    make_estimator: Callable[[], PassAtKEstimator],
    ks: Sequence[int],
    scheme: Scheme = "iid",
    clusters: np.ndarray | None = None,
    block: int = 10,
    n_boot: int = 500,
    level: float = 0.95,
    seed: int = 0,
) -> Interval:
    a, b = np.asarray(counts_a), np.asarray(counts_b)
    if a.shape != b.shape:
        raise ValueError("paired comparison needs the same problems for both models")
    ks_arr = np.asarray(ks, int)

    def diff(idx: np.ndarray | slice) -> np.ndarray:
        return make_estimator().fit(b[idx], n).pass_at_k(ks_arr) - make_estimator().fit(a[idx], n).pass_at_k(ks_arr)

    est = diff(slice(None))
    rng = np.random.default_rng(seed)
    reps = np.empty((n_boot, ks_arr.size))
    for i in range(n_boot):
        reps[i] = diff(_resample_indices(rng, a.size, scheme, clusters, block))
    alpha = (1 - level) / 2
    lo, hi = np.nanquantile(reps, [alpha, 1 - alpha], axis=0)
    return Interval(ks_arr, est, lo, hi, np.nanstd(reps, axis=0, ddof=1), f"paired-{scheme}", level)
