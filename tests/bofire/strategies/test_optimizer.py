import pytest

from bofire.benchmarks import api as benchmarks
from bofire.data_models.strategies import api as data_models_strategies

from bofire.strategies import api as strategies
from bofire.strategies.predictives.acqf_optimization import get_optimizer

@pytest.fixture(params=[
    ("Himmelblau", {}),
    ("DTLZ2", dict(dim=2, num_objectives=2)),
    ("Ackley", dict(num_categories=3, categorical=True, dim=4)),
])
def benchmark(request):
    benchmark_name, params = request.param
    return getattr(benchmarks, benchmark_name)(**params)

@pytest.fixture()
def optimization_scope(benchmark):
    """ """
    domain = benchmark.domain

    strategy_data = data_models_strategies.SoboStrategy(domain=domain)
    strategy = strategies.SoboStrategy(data_model=strategy_data)

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
