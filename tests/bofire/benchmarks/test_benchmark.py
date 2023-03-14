import pandas as pd

import bofire.benchmarks.benchmark as benchmark
from bofire.benchmarks.multi import ZDT1
from bofire.data_models.domain.api import Domain
from bofire.data_models.strategies.api import (
    RejectionSampler as RejectionSamplerDataModel,
)
from bofire.strategies.api import QparegoStrategy, RejectionSampler
from bofire.strategies.multiobjective import compute_hypervolume

# from bofire.data_models.strategies.api import QparegoStrategy


# TODO: re-enable this benchmark test
def _test_benchmark():
    zdt1 = ZDT1(n_inputs=5)
    qparego_factory = QparegoStrategy

    n_initial_samples = 10
    n_runs = 3
    n_iterations = 2

    def sample(domain):
        nonlocal n_initial_samples
        sampler = RejectionSampler(data_model=RejectionSamplerDataModel(domain=domain))
        sampled = sampler.ask(n_initial_samples)

        return sampled

    def hypervolume(domain: Domain) -> float:
        assert domain.experiments is not None
        return compute_hypervolume(
            domain, domain.experiments, ref_point={"y1": 10, "y2": 10}
        )

    results = benchmark.run(
        zdt1,
        strategy_factory=qparego_factory,
        n_iterations=n_iterations,
        metric=hypervolume,
        initial_sampler=sample,
        n_runs=n_runs,
        n_procs=1,
    )

    assert len(results) == n_runs
    for bench, best in results:
        assert bench.experiments is not None
        assert bench.experiments.shape[0] == n_initial_samples + n_iterations
        assert best.shape[0] == n_iterations
        assert isinstance(best, pd.Series)
