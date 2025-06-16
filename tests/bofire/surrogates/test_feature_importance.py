import numpy as np
import pandas as pd
import pytest
import shap

import bofire.surrogates.api as surrogates
from bofire.benchmarks.api import DTLZ2
from bofire.data_models.domain.api import Inputs, Outputs
from bofire.data_models.features.api import ContinuousInput, ContinuousOutput
from bofire.data_models.kernels.api import RBFKernel, ScaleKernel
from bofire.data_models.surrogates.api import SingleTaskGPSurrogate
from bofire.strategies.api import MoboStrategy
from bofire.surrogates.diagnostics import metrics
from bofire.surrogates.feature_importance import (
    combine_lengthscale_importances,
    combine_permutation_importances,
    combine_shap_importances,
    lengthscale_importance,
    lengthscale_importance_hook,
    permutation_importance,
    permutation_importance_hook,
    shap_importance,
    shap_importance_hook,
)


def get_model_and_data():
    inputs = Inputs(
        features=[
            ContinuousInput(
                key=f"x_{i+1}",
                bounds=(-4, 4),
            )
            for i in range(3)
        ],
    )
    outputs = Outputs(features=[ContinuousOutput(key="y")])
    experiments = inputs.sample(n=20)
    experiments.eval("y=((x_1**2 + x_2 - 11)**2+(x_1 + x_2**2 -7)**2)", inplace=True)
    experiments["valid_y"] = 1
    model = SingleTaskGPSurrogate(
        inputs=inputs,
        outputs=outputs,
    )
    model = surrogates.map(model)
    return model, experiments


def test_shap_importance_for_surrogate():
    model, experiments = get_model_and_data()
    surrogate_data = SingleTaskGPSurrogate(
        inputs=model.inputs,
        outputs=model.outputs,
    )
    surrogate = surrogates.map(surrogate_data)
    surrogate.fit(experiments)
    importance = shap_importance(
        predictor=surrogate, experiments=experiments, bg_experiments=experiments
    )
    assert sorted(importance.keys()) == sorted(surrogate.outputs.get_keys())
    assert isinstance(importance["y"], shap.Explanation)
    # now we test the hook
    X = experiments[model.inputs.get_keys()]
    y = experiments[["y"]]
    importance = shap_importance_hook(
        surrogate=surrogate, X_train=X, y_train=y, X_test=X, y_test=y
    )
    assert sorted(importance.keys()) == sorted(surrogate.outputs.get_keys())
    assert isinstance(importance["y"], shap.Explanation)


def test_shap_importance_for_predictive_strategy():
    # we use here a MO benchmark to check that it works also over multiple outputs
    bench = DTLZ2(dim=6)
    experiments = bench.f(bench.domain.inputs.sample(n=10), return_complete=True)
    candidates = bench.domain.inputs.sample(n=4)
    strategy = MoboStrategy.make(domain=bench.domain)
    strategy.tell(experiments=experiments)
    importance = shap_importance(
        predictor=strategy, bg_experiments=strategy.experiments, experiments=candidates
    )
    assert sorted(importance.keys()) == sorted(strategy.domain.outputs.get_keys())
    for key in importance.keys():
        assert isinstance(importance[key], shap.Explanation)
        assert len(importance[key].values) == len(candidates)
        assert len(importance[key].data) == len(candidates)
        assert len(importance[key].feature_names) == len(
            strategy.domain.inputs.get_keys()
        )


def test_combine_shap_importances():
    model, experiments = get_model_and_data()
    surrogate_data = SingleTaskGPSurrogate(
        inputs=model.inputs,
        outputs=model.outputs,
    )
    surrogate = surrogates.map(surrogate_data)
    _, _, pi = surrogate.cross_validate(
        experiments=experiments,
        folds=3,
        hooks={"shap_importance": shap_importance_hook},
    )

    assert isinstance(pi["shap_importance"], list)
    assert len(pi["shap_importance"]) == 3
    combined = combine_shap_importances(
        shap_values=pi["shap_importance"],
    )
    assert isinstance(combined, dict)
    assert isinstance(combined["y"], shap.Explanation)
    assert len(combined["y"].values) == len(experiments)
    assert len(combined["y"].data) == len(experiments)
    assert len(combined["y"].feature_names) == len(surrogate.inputs.get_keys())


def test_lengthscale_importance_invalid():
    model, experiments = get_model_and_data()
    for kernel in [ScaleKernel(base_kernel=RBFKernel(ard=False)), RBFKernel(ard=False)]:
        surrogate_data = SingleTaskGPSurrogate(
            inputs=model.inputs,
            outputs=model.outputs,
            kernel=kernel,
        )
        surrogate = surrogates.map(surrogate_data)
        surrogate.fit(experiments)
        with pytest.raises(
            ValueError, match="Only one lengthscale found, use `ard=True`."
        ):
            lengthscale_importance(surrogate=surrogate)


def test_lengthscale_importance():
    model, experiments = get_model_and_data()
    for kernel in [ScaleKernel(base_kernel=RBFKernel()), RBFKernel()]:
        surrogate_data = SingleTaskGPSurrogate(
            inputs=model.inputs, outputs=model.outputs, kernel=kernel
        )
        surrogate = surrogates.map(surrogate_data)
        surrogate.fit(experiments)
        importance = lengthscale_importance(surrogate=surrogate)
        assert isinstance(importance, pd.Series)
        assert list(importance.index) == surrogate.inputs.get_keys()
        importance = lengthscale_importance_hook(surrogate=surrogate)
        assert isinstance(importance, pd.Series)
        assert list(importance.index) == surrogate.inputs.get_keys()


def test_combine_lengthscale_importances():
    importances = [
        pd.Series(index=["x_1", "x_2", "x_3"], data=np.random.uniform(size=3))
        for _ in range(5)
    ]
    combined = combine_lengthscale_importances(importances=importances)
    assert isinstance(combined, pd.DataFrame)
    assert combined.shape == (5, 3)
    assert list(combined.columns) == ["x_1", "x_2", "x_3"]


def test_permutation_importance_invalid():
    model, experiments = get_model_and_data()
    X = experiments[model.inputs.get_keys()]
    y = experiments[["y"]]
    model.fit(experiments=experiments)
    with pytest.raises(AssertionError):
        permutation_importance(surrogate=model, X=X, y=y, n_repeats=1)
    with pytest.raises(AssertionError):
        permutation_importance(surrogate=model, X=X, y=y, n_repeats=2, seed=-1)


def test_permutation_importance():
    model, experiments = get_model_and_data()
    X = experiments[model.inputs.get_keys()]
    y = experiments[["y"]]
    model.fit(experiments=experiments)
    results = permutation_importance(surrogate=model, X=X, y=y, n_repeats=5)
    assert isinstance(results, dict)
    assert len(results) == len(metrics)
    for m in metrics.keys():
        assert m.name in results.keys()
        assert isinstance(results[m.name], pd.DataFrame)
        assert list(results[m.name].columns) == model.inputs.get_keys()
        assert list(results[m.name].index) == ["mean", "std"]


def test_permutation_importance_nan():
    model, experiments = get_model_and_data()
    X = experiments[model.inputs.get_keys()][:1]
    y = experiments[["y"]][:1]
    model.fit(experiments=experiments)
    results = permutation_importance(surrogate=model, X=X, y=y, n_repeats=5)
    assert isinstance(results, dict)
    assert len(results) == len(metrics)
    for m in metrics.keys():
        assert m.name in results.keys()
        assert isinstance(results[m.name], pd.DataFrame)
        assert list(results[m.name].columns) == model.inputs.get_keys()
        assert list(results[m.name].index) == ["mean", "std"]
        assert len(results[m.name].dropna()) == 0


@pytest.mark.parametrize("use_test", [True, False])
def test_permutation_importance_hook(use_test):
    model, experiments = get_model_and_data()
    X = experiments[model.inputs.get_keys()]
    y = experiments[["y"]]
    model.fit(experiments=experiments)
    results = permutation_importance_hook(
        surrogate=model,
        X_train=X,
        y_train=y,
        X_test=X,
        y_test=y,
        use_test=use_test,
    )
    assert isinstance(results, dict)
    assert len(results) == len(metrics)
    for m in metrics.keys():
        assert m.name in results.keys()
        assert isinstance(results[m.name], pd.DataFrame)
        assert list(results[m.name].columns) == model.inputs.get_keys()
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
            importances=pi["pemutation_importance"],
            metric=m,
        )
        assert list(importance.columns) == model.inputs.get_keys()
        assert len(importance) == n_folds
