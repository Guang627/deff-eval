"""Unbiased estimator of Chen et al. (2021), "Evaluating Large Language Models
Trained on Code". It is the plug-in of the *empirical* count distribution, so
it is unbiased for k <= n and undefined for k > n."""

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
