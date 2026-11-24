"""passk-smooth: code for "pilot-predicted design effects for pass@k" (GSM-Symbolic, SWE-bench Verified, MATH-500).

The package holds the generation pipeline (vLLM sampling + answer grading -> per-problem success counts), the unbiased
pass@k estimator and interval helpers. The analyses in the paper live in scripts/analysis/.
"""

from .estimators import UnbiasedEstimator, passk_weights
from .inference import bootstrap_ci, cluster_robust_se, design_effect, wilson_interval

__version__ = "0.2.0"
__all__ = ["UnbiasedEstimator", "bootstrap_ci", "cluster_robust_se", "design_effect", "passk_weights", "wilson_interval"]
