import pandas as pd

import bofire.strategies.api as strategies
import bofire.strategies.mapper as strategy_mapper
from bofire.benchmarks.multi import ZDT1
from bofire.data_models.domain.api import Domain
from bofire.data_models.strategies.api import (
    QparegoStrategy as QparegoStrategyDataModel,
)
from bofire.data_models.strategies.api import RandomStrategy as RandomStrategyDataModel
from bofire.runners.api import run
from bofire.utils.multiobjective import compute_hypervolume


def test_benchmark():
    zdt1 = ZDT1(n_inputs=5)
    qparego_factory = QparegoStrategyDataModel

    n_initial_samples = 10
    n_runs = 3
    n_iterations = 2

    def sample(domain):
        nonlocal n_initial_samples
        sampler = strategies.map(RandomStrategyDataModel(domain=domain))
        sampled = sampler.ask(n_initial_samples)

        return sampled

    def hypervolume(domain: Domain, experiments: pd.DataFrame) -> float:
        return compute_hypervolume(domain, experiments, ref_point={"y1": 10, "y2": 10})

    results = run(
        zdt1,
        strategy_factory=lambda domain: strategy_mapper.map(
            qparego_factory(domain=domain),
        ),
        n_iterations=n_iterations,
        metric=hypervolume,
        initial_sampler=sample,
        n_runs=n_runs,
        n_procs=1,
    )

    assert len(results) == n_runs
    for experiments, best in results:
        assert experiments is not None
        assert experiments.shape[0] == n_initial_samples + n_iterations
        assert best.shape[0] == n_iterations
        assert isinstance(best, pd.Series)
        assert isinstance(experiments, pd.DataFrame)
