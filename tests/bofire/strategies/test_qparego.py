import random
from itertools import chain

import pytest
import torch

import bofire.data_models.strategies.api as data_models
import bofire.data_models.surrogates.api as surrogate_data_models
import tests.bofire.data_models.specs.api as specs
from bofire.benchmarks.multi import C2DTLZ2, DTLZ2, CrossCoupling
from bofire.data_models.domain.api import Outputs
from bofire.data_models.strategies.api import (
    PolytopeSampler as PolytopeSamplerDataModel,
)
from bofire.strategies.api import PolytopeSampler, QparegoStrategy
from tests.bofire.strategies.test_base import domains
from tests.bofire.strategies.test_multiobjective import invalid_domains

VALID_BOTORCH_QPAREGO_STRATEGY_SPEC = {
    "domain": domains[6],
    "surrogate_specs": surrogate_data_models.BotorchSurrogates(
        surrogates=[
            surrogate_data_models.SingleTaskGPSurrogate(
                input_features=domains[6].input_features,
                output_features=Outputs(
                    features=[domains[6].output_features.get_by_key("of1")]
                ),
            ),
            surrogate_data_models.SingleTaskGPSurrogate(
                input_features=domains[6].input_features,
                output_features=Outputs(
                    features=[domains[6].output_features.get_by_key("of2")]
                ),
            ),
        ]
    ),
    "descriptor_method": "FREE",
    "acquisition_function": specs.acquisition_functions.valid().obj(),
    "categorical_method": "FREE",
}

# BotorchSurrogates(
#                models=[
#                    SingleTaskGPSurrogate(
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
            "surrogate_specs": surrogate_data_models.BotorchSurrogates(
                surrogates=[
                    surrogate_data_models.MixedSingleTaskGPSurrogate(
                        input_features=domains[2].input_features,
                        output_features=Outputs(
                            features=[domains[2].output_features.get_by_key("of1")]
                        ),
                        constraints=[],
                    ),
                    surrogate_data_models.MixedSingleTaskGPSurrogate(
                        input_features=domains[2].input_features,
                        output_features=Outputs(
                            features=[domains[2].output_features.get_by_key("of2")]
                        ),
                        constraints=[],
                    ),
                ]
            ),
            "descriptor_method": "EXHAUSTIVE",
            "categorical_method": "EXHAUSTIVE",
        },
        {**VALID_BOTORCH_QPAREGO_STRATEGY_SPEC, "seed": 1},
        # {**VALID_BOTORCH_QPAREGO_STRATEGY_SPEC, "surrogate_specs": VALID_MODEL_SPEC_LIST},
    ],
    "invalids": [
        {**VALID_BOTORCH_QPAREGO_STRATEGY_SPEC, "descriptor_method": None},
        {**VALID_BOTORCH_QPAREGO_STRATEGY_SPEC, "categorical_method": None},
        # {**VALID_BOTORCH_QPAREGO_STRATEGY_SPEC, "seed": -1},
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
    data_model = data_models.QparegoStrategy(domain=domain)
    with pytest.raises(ValueError):
        QparegoStrategy(data_model=data_model)


@pytest.mark.parametrize(
    "num_test_candidates",
    [num_test_candidates for num_test_candidates in range(1, 2)],
)
def test_qparego(num_test_candidates):
    # generate data
    benchmark = DTLZ2(dim=6)
    random_strategy = PolytopeSampler(
        data_model=PolytopeSamplerDataModel(domain=benchmark.domain)
    )
    experiments = benchmark.f(random_strategy._ask(n=10), return_complete=True)
    # init strategy
    data_model = data_models.QparegoStrategy(domain=benchmark.domain)
    my_strategy = QparegoStrategy(data_model=data_model)
    my_strategy.tell(experiments)
    # ask
    candidates = my_strategy.ask(num_test_candidates)
    assert len(candidates) == num_test_candidates


@pytest.mark.parametrize(
    "num_test_candidates",
    [num_test_candidates for num_test_candidates in range(1, 2)],
)
def test_qparego_constraints(num_test_candidates):
    # generate data
    benchmark = C2DTLZ2(dim=4)
    random_strategy = PolytopeSampler(
        data_model=PolytopeSamplerDataModel(domain=benchmark.domain)
    )
    experiments = benchmark.f(random_strategy._ask(n=10), return_complete=True)
    # init strategy
    data_model = data_models.QparegoStrategy(domain=benchmark.domain)
    my_strategy = QparegoStrategy(data_model=data_model)
    my_strategy.tell(experiments)
    # ask
    candidates = my_strategy.ask(num_test_candidates)
    assert len(candidates) == num_test_candidates


@pytest.mark.parametrize(
    "specs, benchmark, num_experiments, num_candidates",
    [
        (
            BOTORCH_QPAREGO_STRATEGY_SPECS["valids"][0],
            DTLZ2(dim=6),
            random.randint(8, 10),
            random.randint(1, 3),
        ),
        (
            BOTORCH_QPAREGO_STRATEGY_SPECS["valids"][1],
            CrossCoupling(),
            random.randint(8, 10),
            random.randint(1, 3),
        ),
    ],
)
@pytest.mark.slow
def test_get_acqf_input(specs, benchmark, num_experiments, num_candidates):
    # generate data
    random_strategy = PolytopeSampler(
        data_model=PolytopeSamplerDataModel(domain=benchmark.domain)
    )
    experiments = benchmark.f(
        random_strategy._ask(n=num_experiments), return_complete=True
    )
    print(specs.items())
    data_model = data_models.QparegoStrategy(
        domain=benchmark.domain,
        **{
            key: value
            for key, value in specs.items()
            if key not in ["domain", "surrogate_specs"]
        }
    )
    strategy = QparegoStrategy(data_model=data_model)
    # just to ensure there are no former experiments/ candidates already stored in the domain

    strategy.tell(experiments)
    strategy.ask(candidate_count=num_candidates, add_pending=True)

    X_train, X_pending = strategy.get_acqf_input_tensors()

    _, names = strategy.domain.input_features._get_transform_info(
        specs=strategy.surrogate_specs.input_preprocessing_specs
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


# def test_qparego_constraints():
#     benchmark = C2DTLZ2(dim=4)
#     random_strategy = PolytopeSampler(domain=benchmark.domain)
#     experiments = benchmark.f(random_strategy._ask(n=10), return_complete=True)
#     my_strategy = QparegoStrategy(domain=benchmark.domain)
#     my_strategy.tell(experiments)
#     assert isinstance(my_strategy.objective, WeightedMCMultiOutputObjective)
#     assert isinstance(my_strategy.acqf, qNoisyExpectedHypervolumeImprovement)
#     assert my_strategy.acqf.eta == torch.tensor(1e-3)
#     assert len(my_strategy.acqf.constraints) == 1
#     assert torch.allclose(
#         my_strategy.acqf.objective.outcomes, torch.tensor([0, 1], dtype=torch.int64)
#     )
#     assert torch.allclose(
#         my_strategy.acqf.objective.weights, torch.tensor([-1, -1], dtype=torch.double)
#     )
#     assert torch.allclose(
#         my_strategy.acqf.ref_point,
#         torch.tensor([-1.1, -1.1], dtype=torch.double),
#     )
