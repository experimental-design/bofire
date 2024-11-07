import numpy as np
from pandas.testing import assert_frame_equal

import bofire.strategies.api as strategies
from bofire.benchmarks import benchmark
from bofire.benchmarks.api import GenericBenchmark
from bofire.benchmarks.multi import ZDT1
from bofire.benchmarks.single import Himmelblau
from bofire.data_models.strategies.api import RandomStrategy


def test_generic_benchmark():
    bench = ZDT1(n_inputs=5)
    genbench = GenericBenchmark(domain=bench.domain, func=bench._f)
    candidates = bench.domain.inputs.sample(5)
    samples1 = bench.f(candidates=candidates)
    samples2 = genbench.f(candidates=candidates)
    assert_frame_equal(samples1, samples2)


def test_benchmark_generate_outliers():
    def sample(domain):
        datamodel = RandomStrategy(domain=domain)
        sampler = strategies.map(data_model=datamodel)
        sampled = sampler.ask(10)
        return sampled

    outlier_rate = 0.5
    Benchmark = Himmelblau()
    sampled = sample(Benchmark.domain)
    sampled_xy = Benchmark.f(sampled, return_complete=True)
    Benchmark = Himmelblau(
        outlier_rate=outlier_rate,
        outlier_prior=benchmark.UniformOutlierPrior(bounds=(50, 100)),
    )
    sampled_xy1 = Benchmark.f(sampled, return_complete=True)
    assert np.sum(sampled_xy["y"] != sampled_xy1["y"]) != 0
    assert isinstance(Benchmark.outlier_prior, benchmark.UniformOutlierPrior)
