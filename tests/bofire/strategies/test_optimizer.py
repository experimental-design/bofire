from typing import Tuple, Callable, List, Type
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pytest

from bofire.benchmarks import api as benchmarks
from bofire.data_models.domain.api import Domain
from bofire.data_models.features.api import ContinuousInput, DiscreteInput
from bofire.data_models.strategies import api as data_models_strategies
from bofire.strategies import api as strategies
from bofire.strategies.predictives.acqf_optimization import get_optimizer, AcquisitionOptimizer

@pytest.fixture(
    params=[ # (optimizer data model, params)
               # ("BotorchOptimizer", {}),
               ("GeneticAlgorithm", {}),
           ])
def optimizer(request) -> data_models_strategies.AcquisitionOptimizer:
    optimizer_str, params = request.param
    return getattr(data_models_strategies, optimizer_str)(**params)


class ContraintCollection:
    @staticmethod
    def nonliner_ineq_for_himmelblau(domain: Domain):
        pass


@dataclass
class OptimizerBenchmark:
    """ collects information for optimization benchmark, excluding the optimizer"""
    benchmark: benchmarks.Benchmark
    n_experiments: int
    strategy: Type[data_models_strategies.Strategy]
    additional_constraint_functions: List[Callable[[Domain], Domain]] = field(default_factory=lambda: {})
    map_conti_inputs_to_discrete: bool = False  # for testing fully categorical problems

    def __call__(self, optimizer: data_models_strategies.AcquisitionOptimizer):
        """ map data-models of strategy and optimizer, tell the predictive strategy"""

        domain = self.benchmark.domain

        if self.map_conti_inputs_to_discrete:
            # replace a continuous input with a discrete input of the same name, but only 5 possible values
            for idx, ft in enumerate(domain.inputs.features):
                if isinstance(ft, ContinuousInput):
                    domain.inputs.features[idx] = DiscreteInput(
                        key=ft.key, values=np.linspace(ft.bounds[0], ft.bounds[1], 5)
                    )

        for f_constr in self.additional_constraint_functions:
            domain = f_constr(domain)

        strategy = self.strategy(domain=domain, acquisition_optimizer=optimizer)
        strategy = strategies.map(strategy)

        experiments = self.benchmark.f(domain.inputs.sample(self.n_experiments), return_complete=True)
        strategy.tell(experiments=experiments)

        input_preprocessing_specs = strategy.input_preprocessing_specs
        acqfs = strategy._get_acqfs(2)

        return acqfs, input_preprocessing_specs, strategy.acqf_optimizer

@pytest.fixture(
    params=[  # (benchmark, n_experiments, params, stategy, map_conti_inputs_to_discrete)
        OptimizerBenchmark(
            benchmarks.Himmelblau(),
            2,
            data_models_strategies.SoboStrategy,
        ),
        OptimizerBenchmark(
            benchmarks.Detergent(), 5, data_models_strategies.AdditiveSoboStrategy,
        ),
        OptimizerBenchmark(
            benchmarks.Detergent(), 5, data_models_strategies.MultiobjectiveStrategy,
        ),
        ("DTLZ2", 3, {"dim": 2, "num_objectives": 2}, "AdditiveSoboStrategy", False),
        (
            "Ackley", 10,
            {"num_categories": 3, "categorical": True, "dim": 4},
            "SoboStrategy",
            False,
        ),
        # ("Detergent", {}, "SoboStrategy", False),
        (
            "Ackley", 10,
            {"num_categories": 3, "categorical": True, "dim": 3},
            "SoboStrategy",
            True,
        ),  # this is for testing the "all-categoric" usecase
    ]
)
def optimizer_benchmark(request) -> OptimizerBenchmark:
    return request.param



def test_optimizer(optimizer_benchmark, optimizer):
    domain, input_preprocessing_specs, experiments, acqfs, optimizer = optimization_scope
    print(domain.constraints)

    candidates, acqf_vals = optimizer.optimize(
        candidate_count=10,
        acqfs=acqfs,
        domain=domain,
        input_preprocessing_specs=input_preprocessing_specs,
        experiments=experiments,
    )

    assert candidates.shape[0] == 10
