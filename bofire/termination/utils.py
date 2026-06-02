"""Pure utility helpers for the termination module."""

from typing import List, Optional

import numpy as np
import pandas as pd


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
) -> Optional[float]:
    """Compute a threshold from cross-validation fold variability.

    Uses the corrected std of the incumbent's per-fold scores
    (C. Nadeau and Y. Bengio, NeurIPS 2003):
    ``threshold = threshold_factor * sqrt(1/K + 1/(K-1)) * std(fold_scores)``.
    The incumbent is the row minimising ``output_key``.

    Args:
        experiments: Experiments conducted so far.
        output_key: Output column used to locate the incumbent.
        cv_fold_columns: Columns containing the per-fold CV scores.
        threshold_factor: Multiplier (``decay`` in Makarova et al. 2022).

    Returns:
        The corrected CV threshold, or ``None`` if fold scores contain NaN
        or have zero variability.
    """
    y_values = experiments[output_key].dropna()
    if len(y_values) < 1:
        return None
    incumbent_idx = y_values.idxmin()
    fold_scores = experiments.loc[incumbent_idx, cv_fold_columns].values.astype(float)
    if np.any(np.isnan(fold_scores)):
        return None
    k = len(cv_fold_columns)
    correction = np.sqrt(1.0 / k + 1.0 / (k - 1))
    fold_std = float(np.std(fold_scores, ddof=0))
    if fold_std <= 0:
        return None
    return float(threshold_factor * correction * fold_std)
