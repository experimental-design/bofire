import pandas as pd

import bofire.benchmarks.benchmark as benchmark
from bofire.benchmarks.multi import ZDT1
from bofire.domain.domain import Domain
from bofire.samplers import RejectionSampler
from bofire.strategies.botorch.qparego import BoTorchQparegoStrategy
from bofire.utils.multiobjective import compute_hypervolume


def test_benchmark():
    zdt1 = ZDT1(n_inputs=5)
    qparego_factory = BoTorchQparegoStrategy

    n_initial_samples = 10
    n_runs = 3
    n_iterations = 2

    def sample(domain):
        nonlocal n_initial_samples
        sampler = RejectionSampler(domain=domain)
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
        assert bench.domain.experiments is not None
        assert bench.domain.experiments.shape[0] == n_initial_samples + n_iterations
        assert best.shape[0] == n_iterations
        assert isinstance(best, pd.Series)
