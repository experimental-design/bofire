import pytest

from bofire.benchmarks.api import Himmelblau, Hyperopt
from bofire.data_models.enum import RegressionMetricsEnum
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
