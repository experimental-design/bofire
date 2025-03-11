import pytest
from typing import Tuple, Type

from bofire.benchmarks import api as benchmarks
from bofire.data_models.strategies import api as data_models_strategies

from bofire.strategies import api as strategies
from bofire.strategies.predictives.acqf_optimization import get_optimizer

@pytest.fixture(params=[
    ("Himmelblau", {}, "SoboStrategy"),
    ("DTLZ2", dict(dim=2, num_objectives=2), "AdditiveSoboStrategy"),
    ("Ackley", dict(num_categories=3, categorical=True, dim=4), "SoboStrategy"),
])
def benchmark(request) -> Tuple[benchmarks.Benchmark, strategies.PredictiveStrategy]:
    benchmark_name, params, strategy = request.param
    bm = getattr(benchmarks, benchmark_name)(**params)
    strategy = getattr(data_models_strategies, strategy)(domain=bm.domain)
    return bm, strategy

@pytest.fixture()
def optimization_scope(benchmark):
    """ """
    benchmark, strategy_data = benchmark
    domain = benchmark.domain

    strategy = strategies.map(strategy_data)

    experiments = benchmark.f(domain.inputs.sample(10), return_complete=True)
    strategy.tell(experiments=experiments)
    input_preprocessing_specs = strategy.input_preprocessing_specs
    acqfs = strategy._get_acqfs(2)

    return domain, input_preprocessing_specs, experiments, acqfs


def test_optimizer(optimization_scope):

    domain, input_preprocessing_specs, experiments, acqfs = optimization_scope

    optimizer_data_model = data_models_strategies.BotorchOptimizer()
    optimizer = get_optimizer(optimizer_data_model)

    candidates, acqf_vals = optimizer.optimize(
        candidate_count=2,
        acqfs=acqfs,
        domain=domain,
        input_preprocessing_specs=input_preprocessing_specs,
        experiments=experiments,
    )
