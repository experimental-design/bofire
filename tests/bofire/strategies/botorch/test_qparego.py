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
from bofire.strategies.botorch.qparego import (
    AcquisitionFunctionEnum,
    BoTorchQparegoStrategy,
)
from tests.bofire.strategies.botorch.test_base import domains
from tests.bofire.strategies.botorch.test_model_spec import VALID_MODEL_SPEC_LIST
from tests.bofire.utils.test_multiobjective import dfs, invalid_domains, valid_domains

VALID_BOTORCH_QPAREGO_STRATEGY_SPEC = {
    "domain": domains[2],
    # "num_sobol_samples": 1024,
    # "num_restarts": 8,
    # "num_raw_samples": 1024,
    "descriptor_encoding": random.choice(list(DescriptorEncodingEnum)),
    "descriptor_method": "FREE",
    "categorical_encoding": "ONE_HOT",
    "base_acquisition_function": random.choice(list(AcquisitionFunctionEnum)),
    "categorical_method": "FREE",
}

BOTORCH_QPAREGO_STRATEGY_SPECS = {
    "valids": [
        VALID_BOTORCH_QPAREGO_STRATEGY_SPEC,
        {**VALID_BOTORCH_QPAREGO_STRATEGY_SPEC, "seed": 1},
        {**VALID_BOTORCH_QPAREGO_STRATEGY_SPEC, "model_specs": VALID_MODEL_SPEC_LIST},
    ],
    "invalids": [
        {**VALID_BOTORCH_QPAREGO_STRATEGY_SPEC, "descriptor_encoding": None},
        {**VALID_BOTORCH_QPAREGO_STRATEGY_SPEC, "categorical_encoding": None},
        {
            **VALID_BOTORCH_QPAREGO_STRATEGY_SPEC,
            "categorical_encoding": "ORDINAL",
            "categorical_method": "FREE",
        },
        {**VALID_BOTORCH_QPAREGO_STRATEGY_SPEC, "seed": -1},
    ],
}


@pytest.mark.parametrize(
    "domain",
    [
        invalid_domains[0],
        invalid_domains[1],
    ],
)
def test_invalid_qparego_init_domain(domain):
    with pytest.raises(ValueError):
        BoTorchQparegoStrategy(domain=domain)


@pytest.mark.parametrize(
    "num_test_candidates, base_acquisition_function",
    [
        (num_test_candidates, base_acquisition_function)
        for num_test_candidates in range(1, 3)
        for base_acquisition_function in list(AcquisitionFunctionEnum)
    ],
)
def test_qparego(num_test_candidates, base_acquisition_function):
    # generate data
    benchmark = DTLZ2(dim=6)
    random_strategy = PolytopeSampler(domain=benchmark.domain)
    experiments = benchmark.run_candidate_experiments(random_strategy._sample(n=10))
    # init strategy
    my_strategy = BoTorchQparegoStrategy(
        domain=benchmark.domain, base_acquisition_function=base_acquisition_function
    )
    my_strategy.tell(experiments)
    # ask
    candidates = my_strategy.ask(num_test_candidates)
    assert len(candidates) == num_test_candidates
