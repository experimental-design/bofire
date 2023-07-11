import pytest

from bofire.benchmarks.api import Himmelblau, Hyperopt, hyperoptimize
from bofire.data_models.enum import RegressionMetricsEnum
from bofire.data_models.kernels.api import MaternKernel, RBFKernel
from bofire.data_models.surrogates.api import SingleTaskGPSurrogate

benchmark = Himmelblau()
experiments = benchmark.f(benchmark.domain.inputs.sample(9), return_complete=True)


def test_Hyperopt_invalid():
    surrogate_data = SingleTaskGPSurrogate(
        inputs=benchmark.domain.inputs,
        outputs=benchmark.domain.outputs,
        hyperconfig=None,
    )
    with pytest.raises(ValueError, match="No hyperoptimization configuration found."):
        Hyperopt(surrogate_data=surrogate_data, training_data=experiments, folds=5)


@pytest.mark.parametrize("n_candidates", [1, 2])
def test_Hyperopt(n_candidates: int):
    surrogate_data = SingleTaskGPSurrogate(
        inputs=benchmark.domain.inputs,
        outputs=benchmark.domain.outputs,
    )
    hy = Hyperopt(surrogate_data=surrogate_data, training_data=experiments, folds=3)
    assert hy.target_metric == surrogate_data.hyperconfig.target_metric
    assert hy.domain == surrogate_data.hyperconfig.domain
    candidates = hy.domain.inputs.sample(n_candidates)
    results = hy.f(candidates=candidates)
    assert len(results) == len(candidates)
    assert f"valid_{hy.target_metric.name}" in results.columns
    for e in RegressionMetricsEnum:
        assert e.name in results.columns
    assert len(results.columns) == 8


def test_hyperoptimize_warning():
    surrogate_data = SingleTaskGPSurrogate(
        inputs=benchmark.domain.inputs,
        outputs=benchmark.domain.outputs,
        hyperconfig=None,
    )
    with pytest.warns(
        match="No hyperopt is possible as no hyperopt config is available. Returning initial config."
    ):
        opt_data, metrics = hyperoptimize(
            surrogate_data=surrogate_data, training_data=experiments, folds=3
        )
    assert opt_data == surrogate_data
    assert len(metrics) == 0
    assert set(metrics.columns) == {e.name for e in RegressionMetricsEnum}


@pytest.mark.parametrize("strategy", ["FactorialStrategy", "RandomStrategy"])
def test_hyperoptimize(strategy: str):
    surrogate_data = SingleTaskGPSurrogate(
        inputs=benchmark.domain.inputs,
        outputs=benchmark.domain.outputs,
    )
    if strategy == "RandomStrategy":
        surrogate_data.hyperconfig.hyperstrategy = strategy
        surrogate_data.hyperconfig.n_iterations = 5

    opt_data, metrics = hyperoptimize(
        surrogate_data=surrogate_data, training_data=experiments, folds=3
    )
    if strategy == "RandomStrategy":
        assert len(metrics) == 5
    else:
        assert len(metrics) == 12

    assert set(metrics.columns) == set(
        [e.name for e in RegressionMetricsEnum]
        + surrogate_data.hyperconfig.domain.inputs.get_keys()
    )
    assert opt_data.kernel.base_kernel.ard == (metrics.iloc[0]["ard"] == "True")
    if metrics.iloc[0].kernel == "matern_1.5":
        assert isinstance(opt_data.kernel.base_kernel, MaternKernel)
        assert opt_data.kernel.base_kernel.nu == 1.5
    elif metrics.iloc[0].kernel == "matern_2.5":
        assert isinstance(opt_data.kernel.base_kernel, MaternKernel)
        assert opt_data.kernel.base_kernel.nu == 2.5
    else:
        assert isinstance(opt_data.kernel.base_kernel, RBFKernel)
