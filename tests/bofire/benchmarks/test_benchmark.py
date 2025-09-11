import numpy as np
from botorch.test_functions.synthetic import Ackley
from pandas.testing import assert_frame_equal

import bofire.strategies.api as strategies
from bofire.benchmarks import benchmark
from bofire.benchmarks.api import GenericBenchmark, SyntheticBoTorch
from bofire.benchmarks.multi import ZDT1
from bofire.benchmarks.single import Himmelblau
from bofire.data_models.objectives.api import MinimizeObjective
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


def test_SyntheticBoTorch():
    test_func = Ackley(dim=4)
    bench = SyntheticBoTorch(test_function=test_func)
    assert len(bench.domain.inputs) == 4
    assert bench.domain.inputs.get_by_key("x_1").bounds == (-32.768, 32.768)
    assert len(bench.domain.outputs) == 1
    assert isinstance(bench.domain.outputs[0].objective, MinimizeObjective)
    experiments = bench.f(bench.domain.inputs.sample(10), return_complete=True)
    assert len(experiments) == 10
    optima = bench.get_optima()
    assert optima.y.to_list()[0] <= 1e-14
