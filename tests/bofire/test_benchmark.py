from functools import partial

import pandas as pd

import bofire.benchmarks.benchmark as benchmark
from bofire.benchmarks.zdt import ZDT1
from bofire.samplers import RejectionSampler
from bofire.strategies.botorch.sobo import BoTorchSoboStrategy, qEI


def test_benchmark():
    zdt1 = ZDT1(n_inputs=5)
    sobo_factory = partial(BoTorchSoboStrategy, acquisition_function=qEI())

    n_initial_samples = 10
    n_runs = 3
    n_iterations = 2

    def sample(domain):
        nonlocal n_initial_samples
        n_initial_samples = n_initial_samples
        sampler = RejectionSampler(domain=domain)
        sampled = sampler.ask(n_initial_samples)
        return sampled

    results = benchmark.run(
        zdt1,
        sobo_factory,
        n_iterations=n_iterations,
        metric=benchmark.best_multiplicative,
        initial_sampler=sample,
        n_runs=n_runs,
    )

    assert len(results) == n_runs
    for bench, best in results:
        assert bench.domain.experiments is not None
        assert bench.domain.experiments.shape[0] == n_initial_samples + n_iterations
        assert best.shape[0] == n_iterations
        assert isinstance(best, pd.Series)


if __name__ == "__main__":
    test_benchmark()
