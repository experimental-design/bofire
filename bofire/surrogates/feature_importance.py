from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from bofire.data_models.enum import RegressionMetricsEnum
from bofire.surrogates.diagnostics import metrics
from bofire.surrogates.single_task_gp import SingleTaskGPSurrogate
from bofire.surrogates.surrogate import Surrogate


def lengthscale_importance(surrogate: SingleTaskGPSurrogate) -> pd.Series:
    """Compute the lengthscale importance based on ARD.

    Args:
        surrogate (SingleTaskGPSurrogate): Surrogate to extract the importances.

    Returns:
        pd.Series: The importance values (inverse of the individual lenght scales).
    """
    try:
        scales = surrogate.model.covar_module.base_kernel.lengthscale  # type: ignore
    except AttributeError:
        raise ValueError("No lenghtscale based kernel found.")
    scales = 1.0 / scales.squeeze().detach().numpy()  # type: ignore
    if isinstance(scales, float):
        raise ValueError("Only one lengthscale found, use `ard=True`.")
    if len(scales) != len(surrogate.inputs):
        raise ValueError(
            "Number of lengthscale parameters to not matches the number of inputs."
        )
    return pd.Series(data=scales, index=surrogate.inputs.get_keys())


def lengthscale_importance_hook(
    surrogate: SingleTaskGPSurrogate,
    X_train: Optional[pd.DataFrame] = None,
    y_train: Optional[pd.DataFrame] = None,
    X_test: Optional[pd.DataFrame] = None,
    y_test: Optional[pd.DataFrame] = None,
):
    """Hook that can be used within `model.cross_validate` to compute a cross validated permutation feature importance."""
    return lengthscale_importance(surrogate=surrogate)


def combine_lengthscale_importances(importances: Sequence[pd.Series]) -> pd.DataFrame:
    """Combine the importance values from each fold into one dataframe.

    Args:
        importances (Sequence[pd.Series]): List of importance values per fold.

    Returns:
        pd.DataFrame: Dataframe with feature keys as columns, and one row per fold.
    """
    return pd.concat(importances, axis=1).T


def permutation_importance(
    surrogate: Surrogate,
    X: pd.DataFrame,
    y: pd.DataFrame,
    n_repeats: int = 5,
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """Computes permutation feature importance for a model.

    Args:
        model (Model): Model for which the feature importances should be estimated.
        X (pd.DataFrame): X values used to estimate the importances.
        y (pd.DataFrame): Y values used to estimate the importances.
        n_repeats (int, optional): Number of repeats. Defaults to 5.
        seed (int, optional): Seed for the random sampler. Defaults to 42.

    Returns:
        Dict[str, pd.DataFrame]: keys are the metrices for which the model is evluated and value is a dataframe
            with the feature keys as columns and the mean and std of the respective permutation importances as rows.
    """
    assert len(surrogate.outputs) == 1, "Only single output model supported so far."
    assert n_repeats > 1, "Number of repeats has to be larger than 1."
    assert seed > 0, "Seed has to be larger than zero."

    signs = {
        RegressionMetricsEnum.R2: 1.0,
        RegressionMetricsEnum.FISHER: -1.0,
        RegressionMetricsEnum.MAE: -1.0,
        RegressionMetricsEnum.MAPE: -1.0,
        RegressionMetricsEnum.MSD: -1.0,
        RegressionMetricsEnum.PEARSON: 1.0,
        RegressionMetricsEnum.SPEARMAN: 1.0,
    }

    output_key = surrogate.outputs[0].key
    rng = np.random.default_rng(seed)
    prelim_results = {
        k.name: {feature.key: [] for feature in surrogate.inputs}
        for k in metrics.keys()
    }
    pred = surrogate.predict(X)
    if len(pred) >= 2:
        original_metrics = {
            k.name: metrics[k](y[output_key].values, pred[output_key + "_pred"].values)  # type: ignore
            for k in metrics.keys()
        }
    else:
        original_metrics = {k.name: np.nan for k in metrics.keys()}

    for feature in surrogate.inputs:
        for _ in range(n_repeats):
            # shuffle
            X_i = X.copy()
            X_i[feature.key] = rng.permutation(X_i[feature.key].values)  # type: ignore
            # predict
            pred = surrogate.predict(X_i)
            # compute scores
            for metricenum, metric in metrics.items():
                if len(pred) >= 2:
                    prelim_results[metricenum.name][feature.key].append(
                        metric(y[output_key].values, pred[output_key + "_pred"].values)  # type: ignore
                    )
                else:
                    prelim_results[metricenum.name][feature.key].append(np.nan)  # type: ignore

    # convert dictionaries to dataframe for easier postprocessing and statistics
    # and return
    results = {}
    for k in metrics.keys():
        results[k.name] = pd.DataFrame(
            data={
                feature.key: [
                    original_metrics[k.name]
                    - np.mean(prelim_results[k.name][feature.key]),
                    np.std(prelim_results[k.name][feature.key]),
                ]
                for feature in surrogate.inputs
            },
            index=["mean", "std"],
        )
        results[k.name].loc["mean"] *= signs[k]

    return results


def permutation_importance_hook(
    surrogate: Surrogate,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    use_test: bool = True,
    n_repeats: int = 5,
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """Hook that can be used within `model.cross_validate` to compute a cross validated permutation feature importance.

    Args:
        model (Model): Predictive BoFire model.
        X_train (pd.DataFrame): Current train fold. X values.
        y_train (pd.DataFrame): Current train fold. y values.
        X_test (pd.DataFrame): Current test fold. X values.
        y_test (pd.DataFrame): Current test fold. y values.
        use_test (bool, optional): If True test fold is used to calculate feature importance else train fold is used.
            Defaults to True.
        n_repeats (int, optional): Number of repeats per feature. Defaults to 5.
        seed (int, optional): Seed for the random number generator. Defaults to 42.

    Returns:
        Dict[str, pd.DataFrame]: keys are the metrices for which the model is evluated and value is a dataframe
            with the feature keys as columns and the mean and std of the respective permutation importances as rows.
    """
    if use_test:
        X = X_test
        y = y_test
    else:
        X = X_train
        y = y_train
    return permutation_importance(
        surrogate=surrogate, X=X, y=y, n_repeats=n_repeats, seed=seed
    )


def combine_permutation_importances(
    importances: Sequence[Dict[str, pd.DataFrame]],
    metric: RegressionMetricsEnum = RegressionMetricsEnum.R2,
) -> pd.DataFrame:
    """Combines feature importances of a set of folds into one data frame for a requested metric.

    Args:
        importances (List[Dict[str, pd.DataFrame]]): List of permutation importance dictionaries, one per fold.
        metric (RegressionMetricsEnum, optional): Metric for which the data should be combined.
            Defaults to RegressionMetricsEnum.R2

    Returns:
        pd.DataFrame: Dataframe holding the mean permutation importance per fold and feature. Can be further processed by
            `describe`.
    """
    feature_keys = importances[0]["MAE"].columns
    return pd.DataFrame(
        data={
            key: [fold[metric.name].loc["mean", key] for fold in importances]
            for key in feature_keys
        }
    )
