from __future__ import annotations

import numpy as np

from .base import PassAtKEstimator, histogram


class UnbiasedEstimator(PassAtKEstimator):
    name = "chen_unbiased"
    extrapolates = False

    def _fit(self, x: np.ndarray, n: int) -> None:
        self._f = histogram(x, n) / x.size

    def pmf(self) -> np.ndarray:
        return self._f

    def _pass_beyond(self, k: int) -> float:
        return float("nan")
