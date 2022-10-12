import random

import numpy as np
import pytest
from botorch.acquisition.multi_objective import (
    qExpectedHypervolumeImprovement, qNoisyExpectedHypervolumeImprovement)
from botorch.acquisition.multi_objective.objective import (
    MCMultiOutputObjective, WeightedMCMultiOutputObjective)
from everest.benchmarks.multiobjective import DTLZ2
from everest.strategies.botorch.base import (CategoricalEncodingEnum,
                                             CategoricalMethodEnum,
                                             DescriptorEncodingEnum,
                                             DescriptorMethodEnum)
from everest.strategies.botorch.qehvi import (BoTorchQehviStrategy,
                                              BoTorchQnehviStrategy)
from everest.strategies.strategy import RandomStrategy
from everest.strategies.tests.test_model_spec import VALID_MODEL_SPEC_LIST
from everest.utils.tests.test_multiobjective import (dfs, invalid_domains,
                                                     valid_domains)

VALID_BOTORCH_QEHVI_STRATEGY_SPEC = {
    # "num_sobol_samples": 1024,
    # "num_restarts": 8,
    # "num_raw_samples": 1024,
    "descriptor_encoding": random.choice(list(DescriptorEncodingEnum)),
    "descriptor_method": random.choice(list(DescriptorMethodEnum)),
    "categorical_encoding": random.choice(list(CategoricalEncodingEnum)),
    "categorical_method": "EXHAUSTIVE",
}

BOTORCH_QEHVI_STRATEGY_SPECS = {
    "valids": [
        VALID_BOTORCH_QEHVI_STRATEGY_SPEC,
        {**VALID_BOTORCH_QEHVI_STRATEGY_SPEC, "seed": 1},
        {**VALID_BOTORCH_QEHVI_STRATEGY_SPEC, "model_specs": VALID_MODEL_SPEC_LIST},
        {**VALID_BOTORCH_QEHVI_STRATEGY_SPEC, "ref_point": {"a":1., "b":2, "c":3}},
    ],
    "invalids": [
        {**VALID_BOTORCH_QEHVI_STRATEGY_SPEC, "descriptor_encoding": None},
        {**VALID_BOTORCH_QEHVI_STRATEGY_SPEC, "categorical_encoding": None},
        {**VALID_BOTORCH_QEHVI_STRATEGY_SPEC, "categorical_encoding": "ORDINAL", "categorical_method": "FREE"},
        {**VALID_BOTORCH_QEHVI_STRATEGY_SPEC, "seed": -1},
    ],
}


@pytest.mark.parametrize("domain, ref_point", [
    (invalid_domains[0], None),
    (invalid_domains[1], None),
    (valid_domains[0], [0])

])
def test_invalid_qehvi_init_domain(domain, ref_point):
    with pytest.raises(ValueError):
        BoTorchQehviStrategy.from_domain(domain, ref_point=ref_point)

@pytest.mark.parametrize("domain, ref_point, experiments, expected", [
    (valid_domains[0], {"of1":0.5,"of2":10.}, dfs[0], [0.5,-10.]),
    (valid_domains[1], {"of1":0.5,"of3":0.5}, dfs[1], [0.5,0.5]),
    (valid_domains[0], None, dfs[0], [1.,-5.]),
    (valid_domains[1], None, dfs[1], [1.,2.])
])
def test_qehvi_get_adjusted_refpoint(domain, ref_point, experiments, expected):
    strategy = BoTorchQehviStrategy.from_domain(domain, ref_point=ref_point)
    # hack for the test to prevent training of the model when using tell
    strategy.experiments = experiments
    adjusted_ref_point = strategy.get_adjusted_refpoint()
    assert isinstance(adjusted_ref_point, list)
    assert np.allclose(expected,np.array(adjusted_ref_point))


@pytest.mark.parametrize("strategy, use_ref_point, num_test_candidates", [
    (strategy, use_ref_point, num_test_candidates)
    for strategy in [BoTorchQehviStrategy,BoTorchQnehviStrategy]
    for use_ref_point in [True, False]
    for num_test_candidates in range(1,3)
])
def test_qehvi(strategy, use_ref_point, num_test_candidates):
    # generate data
    benchmark = DTLZ2(dim=6)
    random_strategy = RandomStrategy.from_domain(benchmark.domain)
    experiments = benchmark.run_candidate_experiments(random_strategy.ask(candidate_count=10)[0])
    experiments_test = benchmark.run_candidate_experiments(random_strategy.ask(candidate_count=num_test_candidates)[0])
    # init strategy
    my_strategy = strategy.from_domain(benchmark.domain, ref_point=benchmark.ref_point if use_ref_point else None)
    my_strategy.tell(experiments)
    assert isinstance(my_strategy.objective, WeightedMCMultiOutputObjective)
    assert isinstance(my_strategy.acqf, qExpectedHypervolumeImprovement if strategy == BoTorchQehviStrategy else qNoisyExpectedHypervolumeImprovement)
    # test acqf calc
    acqf_vals = my_strategy.calc_acquisition(experiments = experiments_test, combined=False)
    assert acqf_vals.shape[0] == num_test_candidates
    acqf_vals = my_strategy.calc_acquisition(experiments = experiments_test, combined=True)
    assert acqf_vals.shape[0] == 1
