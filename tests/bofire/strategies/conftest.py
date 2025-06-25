from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable, List, Type

import numpy as np
import pytest

from bofire.benchmarks import api as benchmarks
from bofire.data_models.constraints import api as constraints_data_models
from bofire.data_models.domain.api import Domain
from bofire.data_models.features.api import ContinuousInput, DiscreteInput
from bofire.data_models.strategies import api as data_models_strategies
from bofire.strategies import api as strategies


class ConstraintCollection:
    @staticmethod
    def constraint_mix_for_himmelblau(domain: Domain) -> Domain:
        domain.constraints.constraints += [
            constraints_data_models.ProductInequalityConstraint(
                features=["x_1", "x_2"],
                exponents=[2, 2],
                rhs=5,
            ),
            constraints_data_models.LinearInequalityConstraint(
                features=["x_1", "x_2"],
                coefficients=[-1.0, 1.0],
                rhs=0.0,
            ),
            constraints_data_models.NonlinearInequalityConstraint(
                expression="x_1**2 + x_2**2 - 5",
                features=["x_1", "x_2"],
                jacobian_expression="2*x_1, 2*x_2",
            ),
        ]
        return domain

    @staticmethod
    def last_input_feature_discrete(domain: Domain) -> Domain:
        """make the first input discrete"""
        domain.inputs.features[-1] = DiscreteInput(
            key=domain.inputs.features[-1].key,
            values=np.linspace(*domain.inputs.features[-1].bounds, 5),
        )
        return domain

    @staticmethod
    def linear_ineq_constr_for_ackley(domain: Domain) -> Domain:
        domain.constraints.constraints += [
            constraints_data_models.LinearInequalityConstraint(
                features=["x_1", "x_2"],
                coefficients=[-1.0, 1.0],
                rhs=0.0,
            ),
        ]
        return domain

    @staticmethod
    def linear_eq_constr_for_ackley(domain: Domain) -> Domain:
        feat = [key for key in domain.inputs.get_keys() if key.startswith("x")]
        domain.constraints.constraints += [
            constraints_data_models.LinearEqualityConstraint(
                features=feat,
                coefficients=[1.0] * len(feat),
                rhs=1.0,
            ),
        ]
        return domain

    @staticmethod
    def nchoosek_constr_for_ackley(domain: Domain) -> Domain:
        for inp in domain.inputs.get(ContinuousInput).features:
            inp.bounds = (0.0, 32.768)
        feat = [key for key in domain.inputs.get_keys() if key.startswith("x")]
        domain.constraints.constraints += [
            constraints_data_models.NChooseKConstraint(
                features=feat,
                min_count=2,
                max_count=3,  # of 4
                none_also_valid=True,
            )
        ]
        return domain

    @staticmethod
    def nchoosek_constr_for_detergent(domain: Domain) -> Domain:
        lb = domain.inputs.get_bounds({})[0]
        feat = [
            key
            for (key, lb_) in zip(domain.inputs.get_keys(), lb)
            if (key.startswith("x") and lb_ == 0.0)
        ]  # leave out x3
        domain.constraints.constraints += [
            constraints_data_models.NChooseKConstraint(
                features=feat,
                min_count=1,
                max_count=2,  # of 4
                none_also_valid=False,
            )
        ]
        return domain


@dataclass
class OptimizerBenchmark:
    """collects information for optimization benchmark, excluding the optimizer"""

    benchmark: benchmarks.Benchmark
    n_experiments: int
    strategy: Type[data_models_strategies.Strategy]
    additional_constraint_functions: List[Callable[[Domain], Domain]] = field(
        default_factory=lambda: {}
    )
    map_conti_inputs_to_discrete: bool = False  # for testing fully categorical problems
    n_add: int = 5

    def get_adapted_domain(self) -> Domain:
        domain = self.benchmark.domain

        if self.map_conti_inputs_to_discrete:
            # replace a continuous input with a discrete input of the same name, but only 5 possible values
            for idx, ft in enumerate(domain.inputs.features):
                if isinstance(ft, ContinuousInput):
                    domain.inputs.features[idx] = DiscreteInput(
                        key=ft.key, values=np.linspace(ft.bounds[0], ft.bounds[1], 5)
                    )

        for f_constr in self.additional_constraint_functions:
            domain = f_constr(deepcopy(domain))

        return domain

    def get_strategy(
        self, optimizer: data_models_strategies.AcquisitionOptimizer
    ) -> strategies.BotorchStrategy:
        """map data-models of strategy and optimizer, tell the predictive strategy"""

        domain = self.get_adapted_domain()

        # check if the optimizer supports nonlinear constraints
        if (
            len(
                domain.constraints.get(
                    [
                        constraints_data_models.NonlinearInequalityConstraint,
                        constraints_data_models.NChooseKConstraint,
                    ]
                ).constraints
            )
            > 0
        ):
            if isinstance(optimizer, data_models_strategies.BotorchOptimizer):
                pytest.skip(
                    "skipping nonlinear constraints and n-choose-k constr. for botorch optimizer"
                )

        strategy = self.strategy(domain=domain, acquisition_optimizer=optimizer)
        strategy = strategies.map(strategy)

        random_strategy = strategies.RandomStrategy.make(domain=domain)
        experiments = self.benchmark.f(
            random_strategy.ask(self.n_experiments), return_complete=True
        )
        strategy.tell(experiments=experiments)

        return strategy


@pytest.fixture(
    params=[
        OptimizerBenchmark(
            benchmarks.Himmelblau(),
            2,
            data_models_strategies.SoboStrategy,
        ),
        OptimizerBenchmark(
            benchmarks.Himmelblau(),
            2,
            data_models_strategies.SoboStrategy,
            additional_constraint_functions=[
                ConstraintCollection.last_input_feature_discrete
            ],
        ),
        OptimizerBenchmark(
            benchmarks.Himmelblau(),
            2,
            data_models_strategies.SoboStrategy,
            additional_constraint_functions=[
                ConstraintCollection.constraint_mix_for_himmelblau
            ],
        ),
        OptimizerBenchmark(
            benchmarks.Detergent(),
            5,
            data_models_strategies.AdditiveSoboStrategy,
        ),
        OptimizerBenchmark(
            benchmarks.Detergent(),
            5,
            data_models_strategies.MultiplicativeSoboStrategy,
            additional_constraint_functions=[
                ConstraintCollection.nchoosek_constr_for_detergent,
            ],
            n_add=10,
        ),
        OptimizerBenchmark(
            benchmarks.CrossCoupling(),
            4,
            data_models_strategies.AdditiveSoboStrategy,
        ),
        OptimizerBenchmark(
            benchmarks.DTLZ2(dim=2, num_objectives=2),
            3,
            data_models_strategies.AdditiveSoboStrategy,
        ),
        OptimizerBenchmark(
            benchmarks.Ackley(num_categories=3, categorical=True, dim=4),
            10,
            data_models_strategies.SoboStrategy,
        ),
        OptimizerBenchmark(
            benchmarks.Ackley(num_categories=3, categorical=True, dim=4),
            10,
            data_models_strategies.SoboStrategy,
            additional_constraint_functions=[
                ConstraintCollection.linear_ineq_constr_for_ackley,
                ConstraintCollection.last_input_feature_discrete,
            ],
        ),
        OptimizerBenchmark(
            benchmarks.Ackley(num_categories=3, categorical=True, dim=4),
            10,
            data_models_strategies.SoboStrategy,
            additional_constraint_functions=[
                ConstraintCollection.linear_ineq_constr_for_ackley,
                ConstraintCollection.linear_eq_constr_for_ackley,
            ],
        ),
        OptimizerBenchmark(
            benchmarks.Ackley(num_categories=3, categorical=True, dim=4),
            10,
            data_models_strategies.SoboStrategy,
            additional_constraint_functions=[
                ConstraintCollection.nchoosek_constr_for_ackley,
            ],
        ),
        OptimizerBenchmark(
            benchmarks.Ackley(num_categories=3, categorical=True, dim=4),
            10,
            data_models_strategies.SoboStrategy,
            additional_constraint_functions=[
                ConstraintCollection.linear_ineq_constr_for_ackley,
                ConstraintCollection.linear_eq_constr_for_ackley,
            ],
        ),
        OptimizerBenchmark(
            benchmarks.Ackley(num_categories=3, categorical=True, dim=4),
            10,
            data_models_strategies.SoboStrategy,
            map_conti_inputs_to_discrete=True,
        ),  # this is for testing the "all-categoric" usecase
    ]
)
def optimizer_benchmark(request) -> OptimizerBenchmark:
    return request.param
