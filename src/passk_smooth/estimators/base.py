from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
from scipy.special import gammaln

ArrayLike = Sequence[int] | np.ndarray


def as_counts(x: ArrayLike, n: int) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("counts must be a non-empty 1-d array")
    if not np.all(np.equal(np.mod(arr, 1), 0)):
        raise ValueError("counts must be integers")
    arr = arr.astype(np.int64)
    if arr.min() < 0 or arr.max() > n:
        raise ValueError(f"counts must lie in [0, {n}]")
    return arr


def histogram(x: np.ndarray, n: int) -> np.ndarray:
    return np.bincount(x, minlength=n + 1).astype(float)


def log_comb(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = np.full(np.broadcast(a, b).shape, -np.inf)
    ok = (b >= 0) & (b <= a)
    a_b, b_b = np.broadcast_arrays(a, b)
    out[ok] = gammaln(a_b[ok] + 1) - gammaln(b_b[ok] + 1) - gammaln(a_b[ok] - b_b[ok] + 1)
    return out


def passk_weights(n: int, k: int) -> np.ndarray:
    if not 1 <= k <= n:
        raise ValueError(f"need 1 <= k <= n, got k={k}, n={n}")
    x = np.arange(n + 1)
    return 1.0 - np.exp(log_comb(n - x, k) - log_comb(n, k))


def binomial_matrix(n: int, p_grid: np.ndarray) -> np.ndarray:
    x = np.arange(n + 1)[:, None]
    p = np.asarray(p_grid, dtype=float)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(p > 0, np.log(np.where(p > 0, p, 1.0)), -np.inf)
        log1p = np.where(p < 1, np.log1p(-np.where(p < 1, p, 0.0)), -np.inf)
        lx = np.where(x > 0, x * logp, 0.0)
        ly = np.where(n - x > 0, (n - x) * log1p, 0.0)
    return np.exp(log_comb(n, x) + lx + ly)


class PassAtKEstimator(ABC):
    name: str = "base"
    extrapolates: bool = True

    def __init__(self) -> None:
        self.n_: int | None = None
        self.m_: int | None = None

    def fit(self, counts: ArrayLike, n: int) -> PassAtKEstimator:
        x = as_counts(counts, n)
        self.n_, self.m_ = n, x.size
        self._fit(x, n)
        return self

    @abstractmethod
    def _fit(self, x: np.ndarray, n: int) -> None: ...

    @abstractmethod
    def pmf(self) -> np.ndarray:
        pass

    @abstractmethod
    def _pass_beyond(self, k: int) -> float:
        pass

    def pass_at_k(self, ks: int | Sequence[int] | np.ndarray) -> np.ndarray:
        if self.n_ is None:
            raise RuntimeError("call fit() first")
        ks_arr = np.atleast_1d(np.asarray(ks, dtype=int))
        f = self.pmf()
        out = np.empty(ks_arr.size)
        for i, k in enumerate(ks_arr):
            if k < 1:
                raise ValueError("k must be >= 1")
            out[i] = float(f @ passk_weights(self.n_, int(k))) if k <= self.n_ else self._pass_beyond(int(k))
        return out

    def params(self) -> dict[str, float]:
        return {}

    def __repr__(self) -> str:  # pragma: no cover
        return f"{type(self).__name__}(name={self.name!r})"
