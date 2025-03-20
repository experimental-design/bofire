import random
from itertools import chain

import pytest
import torch
from botorch.acquisition import (
    qExpectedImprovement,
    qLogExpectedImprovement,
    qLogNoisyExpectedImprovement,
    qNoisyExpectedImprovement,
)
from botorch.acquisition.objective import GenericMCObjective
from pydantic import ValidationError

import bofire.data_models.strategies.api as data_models
from bofire.benchmarks.detergent import Detergent
from bofire.benchmarks.multi import C2DTLZ2, DTLZ2, CrossCoupling
from bofire.data_models.acquisition_functions.api import qEI, qLogEI, qLogNEI, qNEI
from bofire.data_models.strategies.api import RandomStrategy as RandomStrategyDataModel
from bofire.strategies.api import QparegoStrategy, RandomStrategy
from tests.bofire.utils.test_multiobjective import invalid_domains


@pytest.mark.parametrize(
    "domain",
    [
        invalid_domains[0],
        invalid_domains[1],
    ],
)
def test_invalid_qparego_init_domain(domain):
    with pytest.raises(ValidationError):
        data_models.QparegoStrategy(domain=domain)


@pytest.mark.parametrize(
    "num_test_candidates",
    list(range(1, 2)),
)
def test_qparego(num_test_candidates):
    # generate data
    benchmark = DTLZ2(dim=6)
    random_strategy = RandomStrategy(
        data_model=RandomStrategyDataModel(domain=benchmark.domain),
    )
    experiments = benchmark.f(random_strategy.ask(10), return_complete=True)
    # init strategy
    acqfs = [qEI(), qLogEI(), qLogNEI(), qNEI()]
    b_acqfs = [
        qExpectedImprovement,
        qLogExpectedImprovement,
        qLogNoisyExpectedImprovement,
        qNoisyExpectedImprovement,
    ]
    i = random.choice([0, 1, 2, 3])

    data_model = data_models.QparegoStrategy(
        domain=benchmark.domain,
        acquisition_function=acqfs[i],
    )
    my_strategy = QparegoStrategy(data_model=data_model)
    my_strategy.tell(experiments)
    # test get objective
    objective, _, _ = my_strategy._get_objective_and_constraints()
    assert isinstance(objective, GenericMCObjective)
    acqfs = my_strategy._get_acqfs(2)
    assert len(acqfs) == 2
    assert isinstance(acqfs[0], b_acqfs[i])
    assert isinstance(acqfs[1], b_acqfs[i])
    # ask
    candidates = my_strategy.ask(num_test_candidates)
    assert len(candidates) == num_test_candidates


@pytest.mark.parametrize(
    "num_test_candidates",
    [1, 2],
)
def test_qparego_constraints(num_test_candidates):
    # generate data
    def test(benchmark_factory):
        benchmark = benchmark_factory()
        random_strategy = RandomStrategy(
            data_model=RandomStrategyDataModel(domain=benchmark.domain),
        )
        experiments = benchmark.f(random_strategy.ask(10), return_complete=True)
        # init strategy
        data_model = data_models.QparegoStrategy(
            domain=benchmark.domain,
            acquisition_optimizer=data_models.BotorchOptimizer(n_restarts=1),
        )
        my_strategy = QparegoStrategy(data_model=data_model)
        my_strategy.tell(experiments)
        # test get objective
        objective, _, _ = my_strategy._get_objective_and_constraints()
        assert isinstance(objective, GenericMCObjective)
        # ask
        candidates = my_strategy.ask(num_test_candidates)
        assert benchmark.domain.constraints.is_fulfilled(candidates).all()
        assert len(candidates) == num_test_candidates

    test(Detergent)
    test(lambda: C2DTLZ2(dim=4))


@pytest.mark.parametrize(
    "benchmark, num_experiments, num_candidates",
    [
        (
            DTLZ2(dim=6),
            random.randint(8, 10),
            random.randint(1, 3),
        ),
        (
            CrossCoupling(),
            random.randint(8, 10),
            random.randint(1, 3),
        ),
    ],
)
@pytest.mark.slow
def test_get_acqf_input(benchmark, num_experiments, num_candidates):
    # generate data
    random_strategy = RandomStrategy(
        data_model=RandomStrategyDataModel(domain=benchmark.domain),
    )
    experiments = benchmark.f(
        random_strategy.ask(num_experiments),
        return_complete=True,
    )
    data_model = data_models.QparegoStrategy(
        domain=benchmark.domain,
    )
    strategy = QparegoStrategy(data_model=data_model)
    # just to ensure there are no former experiments/ candidates already stored in the domain

    strategy.tell(experiments)
    strategy.ask(candidate_count=num_candidates, add_pending=True)

    X_train, X_pending = strategy.get_acqf_input_tensors()

    _, names = strategy.domain.inputs._get_transform_info(
        specs=strategy.surrogate_specs.input_preprocessing_specs,
    )

    assert torch.is_tensor(X_train)
    assert torch.is_tensor(X_pending)
    assert X_train.shape == (
        num_experiments,
        len(set(chain(*names.values()))),
    )
    assert X_pending.shape == (  # type: ignore
        num_candidates,
        len(set(chain(*names.values()))),
    )
