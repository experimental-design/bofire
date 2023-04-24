from typing import Dict, Sequence

import numpy as np
import pandas as pd

from bofire.data_models.enum import RegressionMetricsEnum
from bofire.surrogates.diagnostics import metrics
from bofire.surrogates.surrogate import Surrogate


def permutation_importance(
    model: Surrogate,
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
    assert len(model.outputs) == 1, "Only single output model supported so far."
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

    output_key = model.outputs[0].key
    rng = np.random.default_rng(seed)
    prelim_results = {
        k.name: {feature.key: [] for feature in model.inputs} for k in metrics.keys()
    }
    pred = model.predict(X)
    original_metrics = {
        k.name: metrics[k](y[output_key].values, pred[output_key + "_pred"].values)  # type: ignore
        for k in metrics.keys()
    }

    for feature in model.inputs:
        for _ in range(n_repeats):
            # shuffle
            X_i = X.copy()
            X_i[feature.key] = rng.permutation(X_i[feature.key].values)  # type: ignore
            # predict
            pred = model.predict(X_i)
            # compute scores
            for metricenum, metric in metrics.items():
                prelim_results[metricenum.name][feature.key].append(
                    metric(y[output_key].values, pred[output_key + "_pred"].values)  # type: ignore
                )
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
                for feature in model.inputs
            },
            index=["mean", "std"],
        )
        results[k.name].loc["mean"] *= signs[k]

    return results


def permutation_importance_hook(
    model: Surrogate,
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
    return permutation_importance(model=model, X=X, y=y, n_repeats=n_repeats, seed=seed)


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
