import numpy as np
import pytest
from scipy.special import comb

from passk_smooth.estimators import UnbiasedEstimator, passk_weights
from passk_smooth.estimators.base import binomial_matrix


def chen_reference(x, n, k):
    """Literal per-problem formula from Chen et al. (2021)."""
    return np.mean([1.0 - comb(n - c, k, exact=True) / comb(n, k, exact=True) for c in x])


@pytest.mark.parametrize("k", [1, 2, 5, 10])
def test_unbiased_matches_reference(counts, k):
    x, _ = counts
    est = UnbiasedEstimator().fit(x, 10)
    assert est.pass_at_k(k)[0] == pytest.approx(chen_reference(x, 10, k), abs=1e-12)


def test_unbiased_is_nan_beyond_n(counts):
    x, _ = counts
    assert np.isnan(UnbiasedEstimator().fit(x, 10).pass_at_k(11)[0])


def test_weights_edge_cases():
    w = passk_weights(10, 10)
    assert w[0] == 0 and np.all(w[1:] == 1)
    np.testing.assert_allclose(passk_weights(10, 1), np.arange(11) / 10)
    with pytest.raises(ValueError):
        passk_weights(5, 6)


def test_binomial_matrix_columns_sum_to_one():
    b = binomial_matrix(7, np.array([0.0, 0.3, 1.0]))
    np.testing.assert_allclose(b.sum(0), 1.0)
    assert b[0, 0] == 1.0 and b[7, 2] == 1.0
