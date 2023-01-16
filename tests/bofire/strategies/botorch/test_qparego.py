import random

import pytest

from bofire.benchmarks.multi import DTLZ2
from bofire.domain.features import OutputFeatures
from bofire.models.torch_models import (
    BotorchModels,
    MixedSingleTaskGPModel,
    SingleTaskGPModel,
)
from bofire.samplers import PolytopeSampler
from bofire.strategies.botorch.qparego import BoTorchQparegoStrategy
from bofire.utils.enum import AcquisitionFunctionEnum
from tests.bofire.strategies.botorch.test_base import domains
from tests.bofire.utils.test_multiobjective import invalid_domains

VALID_BOTORCH_QPAREGO_STRATEGY_SPEC = {
    "domain": domains[6],
    "model_specs": BotorchModels(
        models=[
            SingleTaskGPModel(
                input_features=domains[6].input_features,
                output_features=OutputFeatures(
                    features=[domains[6].output_features.get_by_key("of1")]
                ),
            ),
            SingleTaskGPModel(
                input_features=domains[6].input_features,
                output_features=OutputFeatures(
                    features=[domains[6].output_features.get_by_key("of2")]
                ),
            ),
        ]
    ),
    "descriptor_method": "FREE",
    "acquisition_function": random.choice(list(AcquisitionFunctionEnum)),
    "categorical_method": "FREE",
}

# BotorchModels(
#                models=[
#                    SingleTaskGPModel(
#                        input_features=domains[1].input_features,
#                        output_features=domains[1].output_features,
#                        input_preprocessing_specs={
#                            "if5": CategoricalEncodingEnum.ONE_HOT,
#                            "if6": CategoricalEncodingEnum.ONE_HOT,
#                        },
#                    )
#                ]
#            ),

BOTORCH_QPAREGO_STRATEGY_SPECS = {
    "valids": [
        VALID_BOTORCH_QPAREGO_STRATEGY_SPEC,
        {
            **VALID_BOTORCH_QPAREGO_STRATEGY_SPEC,
            "domain": domains[2],
            "model_specs": BotorchModels(
                models=[
                    MixedSingleTaskGPModel(
                        input_features=domains[2].input_features,
                        output_features=OutputFeatures(
                            features=[domains[2].output_features.get_by_key("of1")]
                        ),
                    ),
                    MixedSingleTaskGPModel(
                        input_features=domains[2].input_features,
                        output_features=OutputFeatures(
                            features=[domains[2].output_features.get_by_key("of2")]
                        ),
                    ),
                ]
            ),
            "descriptor_method": "EXHAUSTIVE",
            "categorical_method": "EXHAUSTIVE",
        },
        {**VALID_BOTORCH_QPAREGO_STRATEGY_SPEC, "seed": 1},
        # {**VALID_BOTORCH_QPAREGO_STRATEGY_SPEC, "model_specs": VALID_MODEL_SPEC_LIST},
    ],
    "invalids": [
        {**VALID_BOTORCH_QPAREGO_STRATEGY_SPEC, "descriptor_method": None},
        {**VALID_BOTORCH_QPAREGO_STRATEGY_SPEC, "categorical_method": None},
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
    "num_test_candidates, acquisition_function",
    [
        (num_test_candidates, acquisition_function)
        for num_test_candidates in range(1, 3)
        for acquisition_function in list(AcquisitionFunctionEnum)
    ],
)
def test_qparego(num_test_candidates, acquisition_function):
    # generate data
    benchmark = DTLZ2(dim=6)
    random_strategy = PolytopeSampler(domain=benchmark.domain)
    experiments = benchmark.f(random_strategy._sample(n=10), return_complete=True)
    # init strategy
    my_strategy = BoTorchQparegoStrategy(
        domain=benchmark.domain, acquisition_function=acquisition_function
    )
    my_strategy.tell(experiments)
    # ask
    candidates = my_strategy.ask(num_test_candidates)
    assert len(candidates) == num_test_candidates


@pytest.mark.parametrize(
    "specs, num_test_candidates",
    [
        (specs, num_test_candidates)
        for specs in BOTORCH_QPAREGO_STRATEGY_SPECS["valids"]
        for num_test_candidates in range(1, 3)
    ],
)
def test_qparego_mixed_input(specs, num_test_candidates):

    my_strategy = BoTorchQparegoStrategy(**specs)
    random_strategy = PolytopeSampler(domain=my_strategy.domain)
    experiments = benchmark.f(random_strategy._sample(n=10), return_complete=True)

    my_strategy.tell(experiments)
    # ask
    candidates = my_strategy.ask(num_test_candidates)
    assert len(candidates) == num_test_candidates
