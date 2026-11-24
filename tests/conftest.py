import os

import numpy as np
import pytest

os.environ.setdefault("PASSK_DISABLE_TRACKING", "1")


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


@pytest.fixture
def counts(rng):
    p = rng.beta(0.5, 0.9, 400)
    p[:60] = 0.0
    p[60:90] = 1.0
    return rng.binomial(10, p), p
