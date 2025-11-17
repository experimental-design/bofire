import numpy as np
import pandas as pd
from botorch.test_functions.synthetic import Ackley
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

import bofire.strategies.api as strategies
from bofire.benchmarks import benchmark
from bofire.benchmarks.api import (
    FormulationWrapper,
    GenericBenchmark,
    SpuriousFeaturesWrapper,
    SyntheticBoTorch,
)
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


def test_FormulationWrapper():
    benchmark = Himmelblau()
    wrapped = FormulationWrapper(benchmark=benchmark, n_filler_features=2)
    assert len(wrapped.domain.inputs) == len(benchmark.domain.inputs) + 2
    assert_array_equal(
        wrapped._mins, np.array([benchmark.domain.inputs[0].bounds[0]] * 2)
    )
    assert_array_equal(
        wrapped._scales, np.array([benchmark.domain.inputs[0].bounds[1] * 2] * 2)
    )
    assert_array_equal(wrapped._scales_new, np.array([0.5, 0.5]))
    # now we test the transform method

    assert len(wrapped.domain.constraints) == 1
    constraint = wrapped.domain.constraints[0]
    assert set(constraint.features) == set(wrapped.domain.inputs.get_keys())
    assert constraint.rhs == 1.0
    assert constraint.coefficients == [1.0] * len(wrapped.domain.inputs)

    candidates = pd.DataFrame(
        {
            "x_1": [0.0, 0.5, 0.25],
            "x_2": [0.0, 0.5, 0.5],
            "x_spurious_0": [1, 0.5, 0.3],
            "x_spurious_1": [0.2, 0.4, 0.3],
        }
    )
    transformed = wrapped._transform(candidates)
    assert transformed.shape == (3, 2)
    assert_frame_equal(
        transformed,
        pd.DataFrame({"x_1": [-6.0, 6.0, 0], "x_2": [-6.0, 6.0, 6.0]}),
    )
    # now we test the full evaluation
    evaled = wrapped.f(candidates, return_complete=False)
    assert_frame_equal(evaled, benchmark.f(transformed, return_complete=False))
    # test adding of NChooseK constraint
    wrapped = FormulationWrapper(benchmark=benchmark, n_filler_features=2, max_count=2)
    assert len(wrapped.domain.constraints) == 2
    nkc = wrapped.domain.constraints[1]
    assert nkc.features == wrapped._benchmark.domain.inputs.get_keys()
    assert nkc.max_count == 2
    assert nkc.min_count == 0
    assert nkc.none_also_valid is True


def test_SpuriousFeaturesWrapper():
    benchmark = Himmelblau()
    wrapped = SpuriousFeaturesWrapper(benchmark=benchmark, n_spurious_features=3)
    assert len(wrapped.domain.inputs) == 5
    # now we test the transform method
    candidates = pd.DataFrame(
        {
            "x_1": [3.0, -2.0, 1.0],
            "x_2": [0.0, 0.5, 5.0],
            "x_spurious_0": [1, 0.5, 0.3],
            "x_spurious_1": [0.2, 0.4, 0.3],
        }
    )
    # now we test the full evaluation
    evaled = wrapped.f(candidates, return_complete=False)
    assert_frame_equal(
        evaled,
        benchmark.f(
            candidates[benchmark.domain.inputs.get_keys()], return_complete=False
        ),
    )
