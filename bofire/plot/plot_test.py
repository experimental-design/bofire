from functools import partial

from bofire.benchmarks.benchmark import run
from bofire.benchmarks.multi import DTLZ2
from bofire.domain.domain import Domain
from bofire.domain.features import (
    ContinuousInput,
    ContinuousOutput,
    InputFeatures,
    OutputFeatures,
)
from bofire.domain.objective import MaximizeObjective
from bofire.plot.feature_importance import plot_feature_importance_by_feature_plotly
from bofire.samplers import PolytopeSampler
from bofire.strategies.botorch.qehvi import BoTorchQehviStrategy
from bofire.strategies.random import RandomStrategy
from bofire.utils.multiobjective import compute_hypervolume

input_features = InputFeatures(
    features=[
        ContinuousInput(key=f"x_{i}", lower_bound=0, upper_bound=1) for i in range(6)
    ]
)
# here the minimize objective is used, if you want to maximize you have to use the maximize objective.
output_features = OutputFeatures(
    features=[
        ContinuousOutput(key=f"f_{i}", objective=MaximizeObjective(w=1.0))
        for i in range(2)
    ]
)
# no constraints are present so we can create the domain
domain = Domain(input_features=input_features, output_features=output_features)


def sample(domain):
    sampler = PolytopeSampler(domain=domain)
    sampled = sampler.ask(10)
    return sampled


def hypervolume(domain: Domain) -> float:
    assert domain.experiments is not None
    return compute_hypervolume(
        domain, domain.experiments, ref_point={"f_0": 1.1, "f_1": 1.1}
    )


random_results = run(
    DTLZ2(dim=6),
    strategy_factory=RandomStrategy,
    n_iterations=50,
    metric=hypervolume,
    # initial_sampler=sample,
    n_runs=1,
    n_procs=1,
)


from bofire.plot.results import plot_scatter_matrix

matrix = plot_scatter_matrix(
    domain=domain,
    experiments=random_results[0][0].domain.experiments,
    objectives=["x_0", "x_1", "f_0", "f_1"],
    display_pareto_only=False,
    diagonal_visible=True,
    showupperhalf=True,
    ref_point={"f_0": 2, "f_1": 2},
    colorstyle="evonik",
)

matrix.show()
