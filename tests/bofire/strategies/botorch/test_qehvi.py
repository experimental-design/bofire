import random

import numpy as np
import pytest
from botorch.acquisition.multi_objective import (
    qExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.objective import (
    MCMultiOutputObjective,
    WeightedMCMultiOutputObjective,
)

from bofire.benchmarks.multiobjective import DTLZ2
from bofire.samplers import PolytopeSampler
from bofire.strategies.botorch.base import (
    CategoricalEncodingEnum,
    CategoricalMethodEnum,
    DescriptorEncodingEnum,
    DescriptorMethodEnum,
)
from bofire.strategies.botorch.qehvi import BoTorchQehviStrategy, BoTorchQnehviStrategy
from tests.bofire.strategies.botorch.test_model_spec import VALID_MODEL_SPEC_LIST
from tests.bofire.utils.test_multiobjective import dfs, invalid_domains, valid_domains

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
        {**VALID_BOTORCH_QEHVI_STRATEGY_SPEC, "ref_point": {"a": 1.0, "b": 2, "c": 3}},
    ],
    "invalids": [
        {**VALID_BOTORCH_QEHVI_STRATEGY_SPEC, "descriptor_encoding": None},
        {**VALID_BOTORCH_QEHVI_STRATEGY_SPEC, "categorical_encoding": None},
        {
            **VALID_BOTORCH_QEHVI_STRATEGY_SPEC,
            "categorical_encoding": "ORDINAL",
            "categorical_method": "FREE",
        },
        {**VALID_BOTORCH_QEHVI_STRATEGY_SPEC, "seed": -1},
    ],
}


@pytest.mark.parametrize(
    "domain, ref_point",
    [(invalid_domains[0], None), (invalid_domains[1], None), (valid_domains[0], [0])],
)
def test_invalid_qehvi_init_domain(domain, ref_point):
    with pytest.raises(ValueError):
        BoTorchQehviStrategy(domain=domain, ref_point=ref_point)


@pytest.mark.parametrize(
    "domain, ref_point, experiments, expected",
    [
        (valid_domains[0], {"of1": 0.5, "of2": 10.0}, dfs[0], [0.5, -10.0]),
        (valid_domains[1], {"of1": 0.5, "of3": 0.5}, dfs[1], [0.5, 0.5]),
        (valid_domains[0], None, dfs[0], [1.0, -5.0]),
        (valid_domains[1], None, dfs[1], [1.0, 2.0]),
    ],
)
def test_qehvi_get_adjusted_refpoint(domain, ref_point, experiments, expected):
    strategy = BoTorchQehviStrategy(domain=domain, ref_point=ref_point)
    # hack for the test to prevent training of the model when using tell
    strategy.domain.set_experiments(experiments)
    adjusted_ref_point = strategy.get_adjusted_refpoint()
    assert isinstance(adjusted_ref_point, list)
    assert np.allclose(expected, np.ndarray(adjusted_ref_point))


@pytest.mark.parametrize(
    "strategy, use_ref_point, num_test_candidates",
    [
        (strategy, use_ref_point, num_test_candidates)
        for strategy in [BoTorchQehviStrategy, BoTorchQnehviStrategy]
        for use_ref_point in [True, False]
        for num_test_candidates in range(1, 3)
    ],
)
def test_qehvi(strategy, use_ref_point, num_test_candidates):
    # generate data
    benchmark = DTLZ2(dim=6)
    random_strategy = PolytopeSampler(domain=benchmark.domain)
    experiments = benchmark.run_candidate_experiments(random_strategy._sample(n=10))
    experiments_test = benchmark.run_candidate_experiments(
        random_strategy._sample(n=num_test_candidates)
    )
    # init strategy
    my_strategy = strategy(
        domain=benchmark.domain,
        ref_point=benchmark.ref_point if use_ref_point else None,
    )
    my_strategy.tell(experiments)
    assert isinstance(my_strategy.objective, WeightedMCMultiOutputObjective)
    assert isinstance(
        my_strategy.acqf,
        qExpectedHypervolumeImprovement
        if strategy == BoTorchQehviStrategy
        else qNoisyExpectedHypervolumeImprovement,
    )
    # test acqf calc
    acqf_vals = my_strategy._choose_from_pool(experiments_test, num_test_candidates)
    assert acqf_vals.shape[0] == num_test_candidates
