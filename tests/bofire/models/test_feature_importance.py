import pandas as pd
import pytest

from bofire.domain.feature import ContinuousInput, ContinuousOutput
from bofire.domain.features import InputFeatures, OutputFeatures
from bofire.models.diagnostics import metrics
from bofire.models.feature_importance import (
    combine_permutation_importances,
    permutation_importance,
    permutation_importance_hook,
)
from bofire.models.gps import SingleTaskGPModel


def get_model_and_data():
    input_features = InputFeatures(
        features=[
            ContinuousInput(key=f"x_{i+1}", lower_bound=-4, upper_bound=4)
            for i in range(3)
        ]
    )
    output_features = OutputFeatures(features=[ContinuousOutput(key="y")])
    experiments = input_features.sample(n=20)
    experiments.eval("y=((x_1**2 + x_2 - 11)**2+(x_1 + x_2**2 -7)**2)", inplace=True)
    experiments["valid_y"] = 1
    model = SingleTaskGPModel(
        input_features=input_features,
        output_features=output_features,
    )
    return model, experiments


def test_permutation_importance_invalid():
    model, experiments = get_model_and_data()
    X = experiments[model.input_features.get_keys()]
    y = experiments[["y"]]
    model.fit(experiments=experiments)
    with pytest.raises(AssertionError):
        permutation_importance(model=model, X=X, y=y, n_repeats=1)
    with pytest.raises(AssertionError):
        permutation_importance(model=model, X=X, y=y, n_repeats=2, seed=-1)


def test_permutation_importance():
    model, experiments = get_model_and_data()
    X = experiments[model.input_features.get_keys()]
    y = experiments[["y"]]
    model.fit(experiments=experiments)
    results = permutation_importance(model=model, X=X, y=y, n_repeats=5)
    assert isinstance(results, dict)
    assert len(results) == len(metrics)
    for m in metrics.keys():
        assert m.name in results.keys()
        assert isinstance(results[m.name], pd.DataFrame)
        assert list(results[m.name].columns) == model.input_features.get_keys()
        assert list(results[m.name].index) == ["mean", "std"]


@pytest.mark.parametrize("use_test", [True, False])
def test_permutation_importance_hook(use_test):
    model, experiments = get_model_and_data()
    X = experiments[model.input_features.get_keys()]
    y = experiments[["y"]]
    model.fit(experiments=experiments)
    results = permutation_importance_hook(
        model=model, X_train=X, y_train=y, X_test=X, y_test=y, use_test=use_test
    )
    assert isinstance(results, dict)
    assert len(results) == len(metrics)
    for m in metrics.keys():
        assert m.name in results.keys()
        assert isinstance(results[m.name], pd.DataFrame)
        assert list(results[m.name].columns) == model.input_features.get_keys()
        assert list(results[m.name].index) == ["mean", "std"]


@pytest.mark.parametrize("n_folds", [5, 3])
def test_combine_permutation_importances(n_folds):
    model, experiments = get_model_and_data()
    _, _, pi = model.cross_validate(
        experiments,
        folds=n_folds,
        hooks={"pemutation_importance": permutation_importance_hook},
    )
    for m in metrics.keys():
        importance = combine_permutation_importances(
            importances=pi["pemutation_importance"], metric=m
        )
        list(importance.columns) == model.input_features.get_keys()
        assert len(importance) == n_folds
