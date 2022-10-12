import pytest
from everest.benchmarks.multiobjective import DTLZ2
from everest.benchmarks.singleobjective import Himmelblau
from everest.strategies.botorch.qehvi import BoTorchQehviStrategy
from everest.strategies.botorch.sobo import BoTorchSoboStrategy
from everest.strategies.strategy import RandomStrategy
from everest.study import MetricsEnum, PoolStudy

NUM_POOL_CANDIDATES = 1000
# setup test case
benchmark = Himmelblau()
random_strategy = RandomStrategy.from_domain(benchmark.domain)
experiments = benchmark.run_candidate_experiments(random_strategy.ask(candidate_count=NUM_POOL_CANDIDATES)[0])

@pytest.mark.parametrize("num_starting_experiments", list(range(1,5)))
def test_generate_uniform(num_starting_experiments):
    idx = PoolStudy.generate_uniform(experiments, num_starting_experiments)
    assert len(idx.shape) == 1
    assert idx.shape[0] == num_starting_experiments

@pytest.mark.parametrize("num_starting_experiments",list(range(5,10)))
def test_poolstudy_picked_open(num_starting_experiments):
    pool = PoolStudy(domain=benchmark.domain,num_starting_experiments=num_starting_experiments,experiments=experiments)
    assert pool.num_picked_experiments == num_starting_experiments
    assert pool.num_open_experiments == NUM_POOL_CANDIDATES - num_starting_experiments
    assert pool.picked_experiments.shape[0] == num_starting_experiments
    assert pool.open_experiments.shape[0] == NUM_POOL_CANDIDATES - num_starting_experiments
    assert pool.meta.iteration.max() == 0

@pytest.mark.parametrize("num_iterations", list(range(1,5)))
def test_poolstudy_optimize(num_iterations):
    num_starting_experiments = 5
    pool = PoolStudy(domain=benchmark.domain,num_starting_experiments=num_starting_experiments,experiments=experiments)
    my_strategy = BoTorchSoboStrategy(acquisition_function='QEI')
    pool.optimize(my_strategy, num_iterations)
    assert pool.num_picked_experiments == num_starting_experiments + num_iterations
    assert pool.num_open_experiments == NUM_POOL_CANDIDATES - num_starting_experiments - num_iterations
    assert pool.picked_experiments.shape[0] == num_starting_experiments + num_iterations
    assert pool.open_experiments.shape[0] == NUM_POOL_CANDIDATES - num_starting_experiments - num_iterations
    assert pool.meta.iteration.max() == num_iterations

def test_expected_random():
    pool = PoolStudy(domain=benchmark.domain,num_starting_experiments=5,experiments=experiments)
    assert pool.expected_random == 500.5

@pytest.mark.parametrize("num_iterations, use_ref_point", [(num_iter, use_ref_point) for num_iter in range(1,3) for use_ref_point in [True, False]])
def test_multiobjectivepoolstudy_optimize(num_iterations, use_ref_point):
    benchmark = DTLZ2(dim=6)
    num_starting_experiments = 10
    random_strategy = RandomStrategy.from_domain(benchmark.domain)
    experiments = benchmark.run_candidate_experiments(random_strategy.ask(candidate_count=NUM_POOL_CANDIDATES)[0])
    #
    pool = PoolStudy(
        domain=benchmark.domain,
        num_starting_experiments=num_starting_experiments,
        experiments=experiments,
        metrics="HYPERVOLUME",
        ref_point=benchmark.ref_point if use_ref_point else None
    )
    my_strategy = BoTorchQehviStrategy(ref_point=benchmark.ref_point)
    pool.optimize(my_strategy,num_iterations=num_iterations)
    assert pool.num_picked_experiments == num_starting_experiments + num_iterations
    assert pool.num_open_experiments == NUM_POOL_CANDIDATES - num_starting_experiments - num_iterations
    assert pool.picked_experiments.shape[0] == num_starting_experiments + num_iterations
    assert pool.open_experiments.shape[0] == NUM_POOL_CANDIDATES - num_starting_experiments - num_iterations
    assert pool.meta.iteration.max() == num_iterations
