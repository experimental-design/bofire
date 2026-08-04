import pytest

from bofire.benchmarks.api import Himmelblau
from bofire.data_models.enum import RegressionMetricsEnum
from bofire.data_models.kernels.api import MaternKernel, RBFKernel
from bofire.data_models.surrogates.api import SingleTaskGPSurrogate
from bofire.runners.api import hyperoptimize


benchmark = Himmelblau()
experiments = benchmark.f(benchmark.domain.inputs.sample(9), return_complete=True)


def test_hyperoptimize_warning():
    surrogate_data = SingleTaskGPSurrogate(
        inputs=benchmark.domain.inputs,
        outputs=benchmark.domain.outputs,
        hyperconfig=None,
    )
    with pytest.warns(
        match="No hyperopt is possible as no hyperopt config is available. Returning initial config.",
    ):
        opt_data, metrics = hyperoptimize(
            surrogate_data=surrogate_data,
            training_data=experiments,
            folds=3,
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
        surrogate_data.hyperconfig.n_iterations = 6

    opt_data, metrics = hyperoptimize(
        surrogate_data=surrogate_data,
        training_data=experiments,
        folds=3,
    )
    if strategy == "RandomStrategy":
        assert len(metrics) == 6
    else:
        assert len(metrics) == 36

    assert set(metrics.columns) == set(
        [e.name for e in RegressionMetricsEnum]
        + surrogate_data.hyperconfig.domain.inputs.get_keys(),
    )
    if hasattr(opt_data.kernel, "base_kernel"):
        assert metrics.iloc[0]["scalekernel"] == "True"
        base_kernel = opt_data.kernel.base_kernel
    else:
        base_kernel = opt_data.kernel
    assert base_kernel.ard == (metrics.iloc[0]["ard"] == "True")
    if metrics.iloc[0].kernel == "matern_1.5":
        assert isinstance(base_kernel, MaternKernel)
        assert base_kernel.nu == 1.5
    elif metrics.iloc[0].kernel == "matern_2.5":
        assert isinstance(base_kernel, MaternKernel)
        assert base_kernel.nu == 2.5
    else:
        assert isinstance(base_kernel, RBFKernel)
