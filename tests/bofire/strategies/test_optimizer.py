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


@pytest.fixture(
    params=[  # (optimizer data model, params)
        # data_models_strategies.BotorchOptimizer(),
        data_models_strategies.GeneticAlgorithm(population_size=100, n_max_gen=100),
    ]
)
def optimizer_data_model(request) -> data_models_strategies.AcquisitionOptimizer:
    return request.param


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
    def linear_constr_for_ackley(domain: Domain) -> Domain:
        feat = [key for key in domain.inputs.get_keys() if key.startswith("x")]
        domain.constraints.constraints += [
            constraints_data_models.LinearEqualityConstraint(
                features=feat,
                coefficients=[1.0] * len(feat),
                rhs=1.0,
            ),
            constraints_data_models.LinearInequalityConstraint(
                features=["x_1", "x_2"],
                coefficients=[-1.0, 1.0],
                rhs=0.0,
            ),
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

    def __call__(
        self, optimizer: data_models_strategies.AcquisitionOptimizer
    ) -> strategies.BotorchStrategy:
        """map data-models of strategy and optimizer, tell the predictive strategy"""

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

        experiments = self.benchmark.f(
            domain.inputs.sample(self.n_experiments), return_complete=True
        )
        strategy.tell(experiments=experiments)

        return strategy


@pytest.fixture(
    params=[
        # OptimizerBenchmark(
        #     benchmarks.Himmelblau(),
        #     2,
        #     data_models_strategies.SoboStrategy,
        # ),
        OptimizerBenchmark(
            benchmarks.Himmelblau(),
            2,
            data_models_strategies.SoboStrategy,
            additional_constraint_functions=[
                ConstraintCollection.constraint_mix_for_himmelblau
            ],
        ),
        # OptimizerBenchmark(
        #     benchmarks.Detergent(),
        #     5,
        #     data_models_strategies.AdditiveSoboStrategy,
        # ),
        # OptimizerBenchmark(
        #     benchmarks.Detergent(),
        #     5,
        #     data_models_strategies.MoboStrategy,
        # ),
        # OptimizerBenchmark(
        #     benchmarks.DTLZ2(dim=2, num_objectives=2),
        #     3,
        #     data_models_strategies.AdditiveSoboStrategy,
        # ),
        # OptimizerBenchmark(
        #     benchmarks.Ackley(num_categories=3, categorical=True, dim=4),
        #     10,
        #     data_models_strategies.SoboStrategy,
        # ),
        OptimizerBenchmark(
            benchmarks.Ackley(num_categories=3, categorical=True, dim=4),
            10,
            data_models_strategies.SoboStrategy,
            additional_constraint_functions=[
                ConstraintCollection.linear_constr_for_ackley
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


def test_optimizer(optimizer_benchmark, optimizer_data_model):
    strategy = optimizer_benchmark(optimizer_data_model)

    proposals = strategy.ask(4)

    assert proposals.shape[0] == 4

    constr = strategy.domain.constraints.get()
    if constr.constraints:
        assert (constr(proposals).values <= 1e-5).all()


def test_linear_projection_repair_function():
    """test the repair function for the linear projection repair function: projecting x_1 / x_2 into the feasible
    space, adhering to a linear constraint x_1 + x_2 = 1 and x_2 <= x_1"""

    optimizer_benchmark = OptimizerBenchmark(
        benchmarks.Ackley(num_categories=3, categorical=True, dim=2),
        4,
        data_models_strategies.SoboStrategy,
        additional_constraint_functions=[ConstraintCollection.linear_constr_for_ackley],
    )
    optimizer_data_model = data_models_strategies.GeneticAlgorithm(
        population_size=100, n_max_gen=100
    )

    strategy = optimizer_benchmark(optimizer_data_model)

    # test the repair function, population size 100, and q=3. Dimension of the problem is 5
    problem, algorithm, termination = (
        strategy.acqf_optimizer._get_problem_and_algorithm(
            strategy.domain,
            strategy.input_preprocessing_specs,
            strategy._get_acqfs(3),
            q=3,
        )
    )
    repair_function = algorithm.repair._do

    sample_population = np.random.uniform(-30, 30, (100, 5 * 3))
    sample_population_repaired = repair_function(problem, sample_population)

    assert sample_population_repaired.shape == sample_population.shape

    def get_x12_points_from_population(X: np.ndarray) -> np.ndarray:
        dims = [np.array([i * 5, 1 + i * 5]) for i in range(3)]
        return np.vstack([X[:, dim] for dim in dims])

    Xpop_c = get_x12_points_from_population(sample_population_repaired)

    # checking constraint adherence (see benchmark constraint functions
    assert (np.abs(Xpop_c.sum(axis=1) - 1.0) < 1e-5).all()
    assert (Xpop_c[:, 0] - Xpop_c[:, 1] > -1e-5).all()
