import random
from itertools import chain

import pytest
import torch

from bofire.benchmarks.multi import DTLZ2
from bofire.domain.features import OutputFeatures
from bofire.models.torch_models import BotorchModels, SingleTaskGPModel
from bofire.samplers import PolytopeSampler
from bofire.strategies.botorch.qparego import BoTorchQparegoStrategy
from bofire.utils.enum import AcquisitionFunctionEnum
from tests.bofire.domain.test_domain_validators import generate_experiments
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
    # "num_sobol_samples": 1024,
    # "num_restarts": 8,
    # "num_raw_samples": 1024,
    # "descriptor_encoding": random.choice(list(DescriptorEncodingEnum)),
    "descriptor_method": "FREE",
    # "categorical_encoding": "ONE_HOT",
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
        # {
        #     **VALID_BOTORCH_QPAREGO_STRATEGY_SPEC,
        #     "domain": domains[2],
        #     "model_specs": BotorchModels(
        #         models=[
        #             MixedSingleTaskGPModel(
        #                 input_features=domains[6].input_features,
        #                 output_features=OutputFeatures(
        #                     features=[domains[6].output_features.get_by_key("of1")]
        #                 ),
        #             ),
        #             MixedSingleTaskGPModel(
        #                 input_features=domains[6].input_features,
        #                 output_features=OutputFeatures(
        #                     features=[domains[6].output_features.get_by_key("of2")]
        #                 ),
        #             ),
        #         ]
        #     ),
        #     "descriptor_method": random.choice(list(CategoricalMethodEnum)),
        #     "categorical_method": random.choice(list(CategoricalMethodEnum)),
        # },
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
    "domain, specs, num_experiments, num_candidates",
    [
        (
            domain,
            random.choice(BOTORCH_QPAREGO_STRATEGY_SPECS["valids"]),
            num_experiments,
            num_candidates,
        )
        for domain in [domains[6]]  # , domains[2]]
        for num_experiments in range(8, 10)
        for num_candidates in range(1, 3)
    ],
)
@pytest.mark.slow
def test_get_acqf_input(domain, specs, num_experiments, num_candidates):

    strategy = BoTorchQparegoStrategy(
        domain=domain, **{key: value for key, value in specs.items() if key != "domain"}
    )

    # just to ensure there are no former experiments/ candidates already stored in the domain
    strategy.domain.experiments = None
    strategy.domain.candidates = None

    experiments = generate_experiments(strategy.domain, num_experiments)
    experiments["if2"][0] = 1
    strategy.tell(experiments)
    strategy.ask(candidate_count=num_candidates, add_pending=True)

    X_train, X_pending = strategy.get_acqf_input_tensors()

    _, names = strategy.domain.input_features._get_transform_info(
        specs=strategy.model_specs.input_preprocessing_specs
    )

    assert torch.is_tensor(X_train)
    assert torch.is_tensor(X_pending)
    assert X_train.shape == (
        num_experiments,
        len(set(chain(*names.values()))),
    )
    assert X_pending.shape == (
        num_candidates,
        len(set(chain(*names.values()))),
    )
