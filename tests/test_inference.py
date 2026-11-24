import numpy as np
import pytest

from passk_smooth.estimators import UnbiasedEstimator, passk_weights
from passk_smooth.inference import bootstrap_ci, cluster_robust_se, design_effect


def test_iid_se_matches_textbook(rng):
    c = rng.normal(size=500)
    assert cluster_robust_se(c) == pytest.approx(c.std(ddof=1) / np.sqrt(500))


def test_cluster_se_inflates_under_cluster_correlation(rng):
    cl = np.repeat(np.arange(25), 20)
    c = rng.normal(size=25)[cl] + 0.3 * rng.normal(size=500)
    assert design_effect(c, cl) > 3.0
    iid = rng.normal(size=500)
    assert design_effect(iid, cl) == pytest.approx(1.0, abs=0.6)


def test_bootstrap_ci_contains_estimate_and_cluster_is_wider(rng):
    cl = np.repeat(np.arange(20), 15)
    p = np.clip(rng.beta(1, 1, 20)[cl] + 0.05 * rng.normal(size=300), 0, 1)
    x = rng.binomial(10, p)
    iid = bootstrap_ci(x, 10, UnbiasedEstimator, [1, 5], "iid", n_boot=200, seed=1)
    clu = bootstrap_ci(x, 10, UnbiasedEstimator, [1, 5], "cluster", clusters=cl, n_boot=200, seed=1)
    assert np.all(iid.lower <= iid.estimate) and np.all(iid.estimate <= iid.upper)
    assert np.all(clu.upper - clu.lower > iid.upper - iid.lower)
    assert iid.estimate[0] == pytest.approx(passk_weights(10, 1)[x].mean())


def test_block_bootstrap_runs(rng):
    x = rng.binomial(5, 0.4, 100)
    ci = bootstrap_ci(x, 5, UnbiasedEstimator, [1], "block", block=7, n_boot=50)
    assert ci.lower[0] < ci.upper[0]


def test_cluster_bootstrap_requires_labels(rng):
    with pytest.raises(ValueError):
        bootstrap_ci(rng.binomial(5, 0.4, 10), 5, UnbiasedEstimator, [1], "cluster", n_boot=5)
