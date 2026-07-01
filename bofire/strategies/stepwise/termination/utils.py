"""Pure utility helpers for the termination module."""

from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.stats import beta as scipy_beta


def compute_threshold_noise(
    noise_variance: Optional[float],
    threshold_factor: float = 1.0,
) -> Optional[float]:
    """Compute ``threshold_factor * noise_variance``.

    Args:
        noise_variance: Observation noise variance, or ``None`` when the
            GP estimate is unavailable.
        threshold_factor: Multiplier applied to ``noise_variance``.

    Returns:
        The threshold, or ``None`` if ``noise_variance`` is unavailable or
        non-positive.
    """
    if noise_variance is None or noise_variance <= 0:
        return None
    return threshold_factor * noise_variance


def compute_threshold_cv(
    experiments: pd.DataFrame,
    output_key: str,
    cv_fold_columns: List[str],
    threshold_factor: float = 1.0,
    sign: float = 1.0,
) -> Optional[float]:
    """Compute a threshold from cross-validation fold variability.

    Uses the corrected std of the incumbent's per-fold scores
    (C. Nadeau and Y. Bengio, NeurIPS 2003):
    ``threshold = threshold_factor * sqrt(1/K + 1/(K-1)) * std(fold_scores)``.
    The incumbent is the row minimising ``sign * output_key``.

    Args:
        experiments: Experiments conducted so far.
        output_key: Output column used to locate the incumbent.
        cv_fold_columns: Columns containing the per-fold CV scores.
        threshold_factor: Multiplier (``decay`` in Makarova et al. 2022).
        sign: ``+1`` (default) when the objective is minimised (incumbent =
            argmin of ``output_key``), ``-1`` when maximised (argmax).

    Returns:
        The corrected CV threshold, or ``None`` if fold scores contain NaN
        or have zero variability.
    """
    y_values = experiments[output_key].dropna()
    if len(y_values) < 1:
        return None
    incumbent_idx = (sign * y_values).idxmin()
    fold_scores = experiments.loc[incumbent_idx, cv_fold_columns].values.astype(float)
    if np.any(np.isnan(fold_scores)):
        return None
    k = len(cv_fold_columns)
    correction = np.sqrt(1.0 / k + 1.0 / (k - 1))
    fold_std = float(np.std(fold_scores, ddof=0))
    if fold_std <= 0:
        return None
    return float(threshold_factor * correction * fold_std)


def clopper_pearson_ci(k: int, n: int, risk: float) -> tuple:
    """Exact Clopper-Pearson confidence interval for a Bernoulli parameter.

    Uses the identity between the binomial CDF and the beta quantile function
    to compute the interval without root-finding.

    Args:
        k: Number of successes out of ``n`` trials.
        n: Total number of trials.
        risk: Total risk level; the interval has coverage ``1 - risk``.

    Returns:
        ``(lower, upper)`` bounds on the Bernoulli parameter.
    """
    half = risk / 2.0
    lower = float(scipy_beta.ppf(half, k, n - k + 1)) if k > 0 else 0.0
    upper = float(scipy_beta.ppf(1.0 - half, k + 1, n - k)) if k < n else 1.0
    return lower, upper
