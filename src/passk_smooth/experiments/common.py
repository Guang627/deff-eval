"""Ground-truth pass@k and attempt subsampling shared by the analysis scripts."""

from __future__ import annotations

import numpy as np
from scipy.stats import hypergeom

from ..estimators.base import log_comb


def true_passk(c: np.ndarray, big_n: int, ks: np.ndarray) -> np.ndarray:
    """Benchmark pass@k from N attempts per problem (Chen et al. at N); NaN for k > N."""
    out = np.full(ks.size, np.nan)
    for i, k in enumerate(ks):
        if k <= big_n:
            out[i] = float(np.mean(1.0 - np.exp(log_comb(big_n - c, k) - log_comb(big_n, k))))
    return out


def true_count_pmf(c: np.ndarray, big_n: int, n: int) -> np.ndarray:
    """Exact pmf of the subsampled count: mixture of Hypergeometric(N, c_i, n)."""
    x = np.arange(n + 1)
    return np.mean([hypergeom.pmf(x, big_n, ci, n) for ci in c], axis=0)


def subsample_counts(rng: np.random.Generator, c: np.ndarray, big_n: int, n: int) -> np.ndarray:
    """Successes among n attempts drawn without replacement from each problem's N."""
    return rng.hypergeometric(c, big_n - c, n)
