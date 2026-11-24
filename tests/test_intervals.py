import numpy as np
import pytest

from passk_smooth.estimators import UnbiasedEstimator
from passk_smooth.inference import paired_bootstrap_diff, wilson_interval


def test_wilson_interval_textbook():
    lo, hi = wilson_interval(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3) and hi == pytest.approx(0.5962, abs=1e-3)
    lo0, hi0 = wilson_interval(0, 20)
    assert lo0 == pytest.approx(0.0, abs=1e-12) and 0 < hi0 < 0.2


def test_paired_bootstrap_identical_models_and_shape_check(counts):
    x, _ = counts
    ci = paired_bootstrap_diff(x, x, 10, UnbiasedEstimator, [1, 10], n_boot=50)
    assert np.allclose(ci.estimate, 0) and np.allclose(ci.lower, 0) and np.allclose(ci.upper, 0)
    with pytest.raises(ValueError):
        paired_bootstrap_diff(x, x[:-1], 10, UnbiasedEstimator, [1], n_boot=5)


def test_paired_is_tighter_than_unpaired(counts, rng):
    x, p = counts
    y = rng.binomial(10, np.clip(p + 0.03, 0, 1))
    ci = paired_bootstrap_diff(x, y, 10, UnbiasedEstimator, [1], n_boot=300, seed=1)
    se_unpaired = np.sqrt(np.var(x / 10, ddof=1) / x.size + np.var(y / 10, ddof=1) / y.size)
    assert ci.se[0] < se_unpaired
