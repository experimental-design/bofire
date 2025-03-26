import itertools
import unittest
import warnings
from typing import Literal, Type

import pandas as pd
import pytest
import torch
from botorch.optim.initializers import gen_batch_initial_conditions

import bofire.data_models.strategies.api as data_models
import bofire.data_models.surrogates.api as surrogate_data_models
import bofire.strategies.api as strategies
import tests.bofire.data_models.specs.api as specs
from bofire.benchmarks.single import Hartmann
from bofire.data_models.constraints.api import (
    Constraint,
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
    ProductInequalityConstraint,
)
from bofire.data_models.domain.api import Domain
from bofire.data_models.enum import CategoricalEncodingEnum, CategoricalMethodEnum
from bofire.data_models.features.api import (
    CategoricalDescriptorInput,
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
    DiscreteInput,
    Feature,
    Output,
)
from bofire.data_models.objectives.api import MaximizeObjective, MinimizeObjective
from bofire.utils.torch_tools import (
    get_nchoosek_constraints,
    get_nonlinear_constraints,
    tkwargs,
)
from tests.bofire.data_models.domain.test_domain_validators import generate_experiments
from tests.bofire.strategies.specs import (
    VALID_ALLOWED_CATEGORICAL_DESCRIPTOR_INPUT_FEATURE_SPEC,
    VALID_CATEGORICAL_DESCRIPTOR_INPUT_FEATURE_SPEC,
    VALID_CATEGORICAL_INPUT_FEATURE_SPEC,
    VALID_CONTINUOUS_INPUT_FEATURE_SPEC,
    VALID_CONTINUOUS_OUTPUT_FEATURE_SPEC,
    VALID_DISCRETE_INPUT_FEATURE_SPEC,
    VALID_FIXED_CATEGORICAL_DESCRIPTOR_INPUT_FEATURE_SPEC,
    VALID_FIXED_CATEGORICAL_INPUT_FEATURE_SPEC,
    VALID_FIXED_CONTINUOUS_INPUT_FEATURE_SPEC,
)


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, append=True)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")


class DummyFeature(Feature):
    type: Literal["DummyFeature"] = "DummyFeature"

    def is_fixed(self):
        pass

    def fixed_value(self):
        pass


class DummyStrategyDataModel(data_models.BotorchStrategy):
    type: Literal["DummyStrategyDataModel"] = "DummyStrategyDataModel"

    def is_constraint_implemented(self, my_type: Type[Constraint]) -> bool:
        return my_type in [
            LinearEqualityConstraint,
            LinearInequalityConstraint,
            NChooseKConstraint,
            ProductInequalityConstraint,
        ]

    @classmethod
    def is_feature_implemented(cls, my_type: Type[Feature]) -> bool:
        return my_type in [
            ContinuousInput,
            CategoricalInput,
            DiscreteInput,
            CategoricalDescriptorInput,
            ContinuousOutput,
        ]

    @classmethod
    def is_objective_implemented(cls, my_type: Type[Feature]) -> bool:
        return my_type in [MinimizeObjective, MaximizeObjective]


class DummyStrategy(strategies.BotorchStrategy):
    def _get_acqfs(
        self,
    ) -> None:
        pass


if1 = ContinuousInput(
    **{
        **VALID_CONTINUOUS_INPUT_FEATURE_SPEC,
        "key": "if1",
    },
)
if2 = ContinuousInput(
    **{
        **VALID_FIXED_CONTINUOUS_INPUT_FEATURE_SPEC,
        "key": "if2",
    },
)

if3 = CategoricalInput(
    **{
        **VALID_CATEGORICAL_INPUT_FEATURE_SPEC,
        "key": "if3",
    },
)

if4 = CategoricalInput(
    **{
        **VALID_FIXED_CATEGORICAL_INPUT_FEATURE_SPEC,
        "key": "if4",
    },
)

if5 = CategoricalDescriptorInput(
    **{
        **VALID_CATEGORICAL_DESCRIPTOR_INPUT_FEATURE_SPEC,
        "key": "if5",
    },
)

if6 = CategoricalDescriptorInput(
    **{
        **VALID_FIXED_CATEGORICAL_DESCRIPTOR_INPUT_FEATURE_SPEC,
        "key": "if6",
    },
)

if7 = DummyFeature(key="if7")

if8 = CategoricalDescriptorInput(
    **{
        **VALID_ALLOWED_CATEGORICAL_DESCRIPTOR_INPUT_FEATURE_SPEC,
        "key": "if8",
    },
)

if9 = DiscreteInput(
    **{
        **VALID_DISCRETE_INPUT_FEATURE_SPEC,
        "key": "if9",
    },
)

of1 = ContinuousOutput(
    **{
        **VALID_CONTINUOUS_OUTPUT_FEATURE_SPEC,
        "key": "of1",
    },
)

of2 = ContinuousOutput(
    **{
        **VALID_CONTINUOUS_OUTPUT_FEATURE_SPEC,
        "key": "of2",
    },
)

domains = [
    Domain.from_lists(
        inputs=[if1, if3, if5, if9],  # no fixed features
        outputs=[of1],
        constraints=[],
    ),
    Domain.from_lists(
        inputs=[
            if1,
            if2,
            if3,
            if4,
            if5,
            if6,
            if9,
        ],  # all feature types incl. with fixed values
        outputs=[of1],
        constraints=[],
    ),
    Domain.from_lists(
        inputs=[
            if1,
            if2,
            if3,
            if4,
            if5,
            if6,
            if9,
        ],  # all feature types incl. with fixed values + mutli-objective
        outputs=[of1, of2],
        constraints=[],
    ),
    Domain.from_lists(
        inputs=[if1, if2],  # only continuous features
        outputs=[of1],
        constraints=[],
    ),
    Domain.from_lists(
        inputs=[if1, if3, if5, if9],  # all feature types + mutli-objective
        outputs=[of1, of2],
        constraints=[],
    ),
    Domain.from_lists(
        inputs=[if1, if2, if8],
        outputs=[of1],
        constraints=[],
    ),
    Domain.from_lists(
        inputs=[if1, if2],  # only continuous features
        outputs=[of1, of2],
        constraints=[],
    ),
    # Domain(
    #     inputs=[if1, if7], # unknown dummy feature
    #     outputs=[of1],
    #     constraints=[],
    # )
]

data = [
    pd.DataFrame.from_dict(
        {
            "if1": [3, 4, 5, 4.5],
            "if3": ["c1", "c2", "c3", "c1"],
            "if5": ["c1", "c2", "c3", "c1"],
            "if9": [1.0, 2.0, 1.0, 2.0],
            "of1": [10, 11, 12, 13],
            "valid_of1": [1, 0, 1, 0],
        },
    ),
    pd.DataFrame.from_dict(
        {
            "if1": [3, 4, 5, 4.5],
            "if2": [3, 3, 3, 3],
            "if3": ["c1", "c2", "c3", "c1"],
            "if4": ["c1", "c1", "c1", "c1"],
            "if5": ["c1", "c2", "c3", "c1"],
            "if6": ["c1", "c1", "c1", "c1"],
            "if9": [1.0, 2.0, 1.0, 2.0],
            "of1": [10, 11, 12, 13],
            "valid_of1": [1, 0, 1, 0],
        },
    ),
    pd.DataFrame.from_dict(
        {
            "if1": [3, 4, 5, 4.5],
            "if2": [3, 3, 3, 3],
            "if3": ["c1", "c2", "c3", "c1"],
            "if4": ["c1", "c1", "c1", "c1"],
            "if5": ["c1", "c2", "c3", "c1"],
            "if6": ["c1", "c1", "c1", "c1"],
            "if9": [1.0, 2.0, 1.0, 2.0],
            "of1": [10, 11, 12, 13],
            "of2": [100, 103, 105, 110],
            "valid_of1": [1, 0, 1, 0],
            "valid_of2": [0, 1, 1, 0],
        },
    ),
    pd.DataFrame.from_dict(
        {
            "if1": [3, 4, 5, 4.5],
            "if2": [3, 3, 3, 3],
            "of1": [10, 11, 12, 13],
            "valid_of1": [1, 0, 1, 0],
        },
    ),
    pd.DataFrame.from_dict(
        {
            "if1": [3, 4, 5, 4.5],
            "if3": ["c1", "c2", "c3", "c1"],
            "if5": ["c1", "c2", "c3", "c1"],
            "if9": [1.0, 2.0, 1.0, 2.0],
            "of1": [10, 11, 12, 13],
            "of2": [100, 103, 105, 110],
            "valid_of1": [1, 0, 1, 0],
            "valid_of2": [0, 1, 1, 0],
        },
    ),
]


@pytest.mark.parametrize("domain", list(domains))
def test_base_create(domain: Domain):
    with pytest.raises(
        ValueError,
        match="Argument is not power of two.",
    ):
        DummyStrategyDataModel(
            domain=domain,
            acquisition_optimizer=data_models.BotorchOptimizer(n_raw_samples=5),
        )


def test_base_invalid_descriptor_method():
    with pytest.raises(ValueError):
        DummyStrategyDataModel(
            domain=domains[0],
            surrogate_specs=[
                surrogate_data_models.SingleTaskGPSurrogate(
                    inputs=domains[0].inputs,
                    outputs=domains[0].outputs,
                    input_preprocessing_specs={"if5": CategoricalEncodingEnum.ONE_HOT},
                ),
            ],
            acquisition_optimizer=data_models.BotorchOptimizer(
                descriptor_method="FREE",
                categorical_method="EXHAUSTIVE",
            ),
        )


@pytest.mark.parametrize(
    # "domain, descriptor_encoding, categorical_encoding, categorical_method, expected",
    "domain, surrogate_specs, categorical_method, descriptor_method, expected",
    [
        (
            domains[0],
            surrogate_data_models.BotorchSurrogates(surrogates=[]),
            "EXHAUSTIVE",
            "EXHAUSTIVE",
            {},
        ),
        (
            domains[1],
            surrogate_data_models.BotorchSurrogates(surrogates=[]),
            "EXHAUSTIVE",
            "EXHAUSTIVE",
            {1: 3, 5: 1, 6: 2, 10: 1, 11: 0, 12: 0},
        ),
        (
            domains[1],
            surrogate_data_models.BotorchSurrogates(
                surrogates=[
                    surrogate_data_models.SingleTaskGPSurrogate(
                        inputs=domains[1].inputs,
                        outputs=domains[1].outputs,
                        input_preprocessing_specs={
                            "if5": CategoricalEncodingEnum.ONE_HOT,
                            "if6": CategoricalEncodingEnum.ONE_HOT,
                        },
                    ),
                ],
            ),
            "EXHAUSTIVE",
            "EXHAUSTIVE",
            {1: 3, 6: 1, 7: 0, 8: 0, 12: 1, 13: 0, 14: 0},
        ),
        (
            domains[1],
            surrogate_data_models.BotorchSurrogates(
                surrogates=[
                    surrogate_data_models.SingleTaskGPSurrogate(
                        inputs=domains[1].inputs,
                        outputs=domains[1].outputs,
                    ),
                ],
            ),
            "FREE",
            "EXHAUSTIVE",
            {1: 3, 5: 1, 6: 2, 10: 1, 11: 0, 12: 0},
        ),
        (
            domains[1],
            surrogate_data_models.BotorchSurrogates(
                surrogates=[
                    surrogate_data_models.SingleTaskGPSurrogate(
                        inputs=domains[1].inputs,
                        outputs=domains[1].outputs,
                        input_preprocessing_specs={
                            "if5": CategoricalEncodingEnum.ONE_HOT,
                            "if6": CategoricalEncodingEnum.ONE_HOT,
                        },
                    ),
                ],
            ),
            "FREE",
            "FREE",
            {1: 3, 6: 1, 7: 0, 8: 0, 12: 1, 13: 0, 14: 0},
        ),
        (
            domains[5],
            surrogate_data_models.BotorchSurrogates(surrogates=[]),
            "EXHAUSTIVE",
            "EXHAUSTIVE",
            {1: 3.0},
        ),
        (
            domains[5],
            surrogate_data_models.BotorchSurrogates(
                surrogates=[
                    surrogate_data_models.SingleTaskGPSurrogate(
                        inputs=domains[5].inputs,
                        outputs=domains[5].outputs,
                        input_preprocessing_specs={
                            "if8": CategoricalEncodingEnum.ONE_HOT,
                        },
                    ),
                ],
            ),
            "FREE",
            "FREE",
            {1: 3.0, 2: 0},
        ),
        (
            domains[5],
            surrogate_data_models.BotorchSurrogates(surrogates=[]),
            "FREE",
            "EXHAUSTIVE",
            {1: 3.0},
        ),
        (
            domains[5],
            surrogate_data_models.BotorchSurrogates(surrogates=[]),
            "FREE",
            "FREE",
            {1: 3.0, 2: 3.0},
        ),
    ],
)
def test_base_get_fixed_features(
    domain,
    surrogate_specs,
    categorical_method,
    descriptor_method,
    expected,
):
    data_model = DummyStrategyDataModel(
        domain=domain,
        surrogate_specs=surrogate_specs,
        acquisition_optimizer=data_models.BotorchOptimizer(
            categorical_method=categorical_method,
            descriptor_method=descriptor_method,
        ),
    )
    myStrategy = DummyStrategy(data_model=data_model)

    experiments = generate_experiments(domain, 100, tol=1.0)
    myStrategy.set_experiments(experiments)

    fixed_features = myStrategy.acqf_optimizer.get_fixed_features(
        domain,
        myStrategy.input_preprocessing_specs,
    )

    assert fixed_features == expected


@pytest.mark.parametrize(
    "domain, descriptor_method, categorical_method, discrete_method, surrogate_specs, expected",
    [
        (
            domains[0],
            "EXHAUSTIVE",
            "EXHAUSTIVE",
            "EXHAUSTIVE",
            surrogate_data_models.BotorchSurrogates(surrogates=[]),
            [
                {4: 1.0, 5: 0.0, 6: 0.0, 2: 1.0, 3: 2.0, 1: 1},
                {4: 1.0, 5: 0.0, 6: 0.0, 2: 3.0, 3: 7.0, 1: 1},
                {4: 1.0, 5: 0.0, 6: 0.0, 2: 5.0, 3: 1.0, 1: 1},
                {4: 0.0, 5: 1.0, 6: 0.0, 2: 1.0, 3: 2.0, 1: 1},
                {4: 0.0, 5: 1.0, 6: 0.0, 2: 3.0, 3: 7.0, 1: 1},
                {4: 0.0, 5: 1.0, 6: 0.0, 2: 5.0, 3: 1.0, 1: 1},
                {4: 0.0, 5: 0.0, 6: 1.0, 2: 1.0, 3: 2.0, 1: 1},
                {4: 0.0, 5: 0.0, 6: 1.0, 2: 3.0, 3: 7.0, 1: 1},
                {4: 0.0, 5: 0.0, 6: 1.0, 2: 5.0, 3: 1.0, 1: 1},
                {4: 1.0, 5: 0.0, 6: 0.0, 2: 1.0, 3: 2.0, 1: 2},
                {4: 1.0, 5: 0.0, 6: 0.0, 2: 3.0, 3: 7.0, 1: 2},
                {4: 1.0, 5: 0.0, 6: 0.0, 2: 5.0, 3: 1.0, 1: 2},
                {4: 0.0, 5: 1.0, 6: 0.0, 2: 1.0, 3: 2.0, 1: 2},
                {4: 0.0, 5: 1.0, 6: 0.0, 2: 3.0, 3: 7.0, 1: 2},
                {4: 0.0, 5: 1.0, 6: 0.0, 2: 5.0, 3: 1.0, 1: 2},
                {4: 0.0, 5: 0.0, 6: 1.0, 2: 1.0, 3: 2.0, 1: 2},
                {4: 0.0, 5: 0.0, 6: 1.0, 2: 3.0, 3: 7.0, 1: 2},
                {4: 0.0, 5: 0.0, 6: 1.0, 2: 5.0, 3: 1.0, 1: 2},
            ],
        ),
        (
            domains[0],
            "EXHAUSTIVE",
            "EXHAUSTIVE",
            "FREE",
            surrogate_data_models.BotorchSurrogates(surrogates=[]),
            [
                {4: 1.0, 5: 0.0, 6: 0.0, 2: 1.0, 3: 2.0},
                {4: 1.0, 5: 0.0, 6: 0.0, 2: 3.0, 3: 7.0},
                {4: 1.0, 5: 0.0, 6: 0.0, 2: 5.0, 3: 1.0},
                {4: 0.0, 5: 1.0, 6: 0.0, 2: 1.0, 3: 2.0},
                {4: 0.0, 5: 1.0, 6: 0.0, 2: 3.0, 3: 7.0},
                {4: 0.0, 5: 1.0, 6: 0.0, 2: 5.0, 3: 1.0},
                {4: 0.0, 5: 0.0, 6: 1.0, 2: 1.0, 3: 2.0},
                {4: 0.0, 5: 0.0, 6: 1.0, 2: 3.0, 3: 7.0},
                {4: 0.0, 5: 0.0, 6: 1.0, 2: 5.0, 3: 1.0},
            ],
        ),
        (
            domains[0],
            "EXHAUSTIVE",
            "EXHAUSTIVE",
            "EXHAUSTIVE",
            surrogate_data_models.BotorchSurrogates(
                surrogates=[
                    surrogate_data_models.SingleTaskGPSurrogate(
                        inputs=domains[0].inputs,
                        outputs=domains[0].outputs,
                        input_preprocessing_specs={
                            "if5": CategoricalEncodingEnum.ONE_HOT,
                        },
                    ),
                ],
            ),
            [
                {2: 1.0, 3: 0.0, 4: 0.0, 5: 1.0, 6: 0.0, 7: 0.0, 1: 1},
                {2: 1.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 1.0, 7: 0.0, 1: 1},
                {2: 1.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 1.0, 1: 1},
                {2: 0.0, 3: 1.0, 4: 0.0, 5: 1.0, 6: 0.0, 7: 0.0, 1: 1},
                {2: 0.0, 3: 1.0, 4: 0.0, 5: 0.0, 6: 1.0, 7: 0.0, 1: 1},
                {2: 0.0, 3: 1.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 1.0, 1: 1},
                {2: 0.0, 3: 0.0, 4: 1.0, 5: 1.0, 6: 0.0, 7: 0.0, 1: 1},
                {2: 0.0, 3: 0.0, 4: 1.0, 5: 0.0, 6: 1.0, 7: 0.0, 1: 1},
                {2: 0.0, 3: 0.0, 4: 1.0, 5: 0.0, 6: 0.0, 7: 1.0, 1: 1},
                {2: 1.0, 3: 0.0, 4: 0.0, 5: 1.0, 6: 0.0, 7: 0.0, 1: 2},
                {2: 1.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 1.0, 7: 0.0, 1: 2},
                {2: 1.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 1.0, 1: 2},
                {2: 0.0, 3: 1.0, 4: 0.0, 5: 1.0, 6: 0.0, 7: 0.0, 1: 2},
                {2: 0.0, 3: 1.0, 4: 0.0, 5: 0.0, 6: 1.0, 7: 0.0, 1: 2},
                {2: 0.0, 3: 1.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 1.0, 1: 2},
                {2: 0.0, 3: 0.0, 4: 1.0, 5: 1.0, 6: 0.0, 7: 0.0, 1: 2},
                {2: 0.0, 3: 0.0, 4: 1.0, 5: 0.0, 6: 1.0, 7: 0.0, 1: 2},
                {2: 0.0, 3: 0.0, 4: 1.0, 5: 0.0, 6: 0.0, 7: 1.0, 1: 2},
            ],
        ),
        (
            domains[0],
            "EXHAUSTIVE",
            "EXHAUSTIVE",
            "FREE",
            surrogate_data_models.BotorchSurrogates(
                surrogates=[
                    surrogate_data_models.SingleTaskGPSurrogate(
                        inputs=domains[0].inputs,
                        outputs=domains[0].outputs,
                        input_preprocessing_specs={
                            "if5": CategoricalEncodingEnum.ONE_HOT,
                        },
                    ),
                ],
            ),
            [
                {2: 1.0, 3: 0.0, 4: 0.0, 5: 1.0, 6: 0.0, 7: 0.0},
                {2: 1.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 1.0, 7: 0.0},
                {2: 1.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 1.0},
                {2: 0.0, 3: 1.0, 4: 0.0, 5: 1.0, 6: 0.0, 7: 0.0},
                {2: 0.0, 3: 1.0, 4: 0.0, 5: 0.0, 6: 1.0, 7: 0.0},
                {2: 0.0, 3: 1.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 1.0},
                {2: 0.0, 3: 0.0, 4: 1.0, 5: 1.0, 6: 0.0, 7: 0.0},
                {2: 0.0, 3: 0.0, 4: 1.0, 5: 0.0, 6: 1.0, 7: 0.0},
                {2: 0.0, 3: 0.0, 4: 1.0, 5: 0.0, 6: 0.0, 7: 1.0},
            ],
        ),
        (
            domains[0],
            "EXHAUSTIVE",
            "FREE",
            "FREE",
            surrogate_data_models.BotorchSurrogates(
                surrogates=[
                    surrogate_data_models.SingleTaskGPSurrogate(
                        inputs=domains[0].inputs,
                        outputs=domains[0].outputs,
                    ),
                ],
            ),
            [{2: 1.0, 3: 2.0}, {2: 3.0, 3: 7.0}, {2: 5.0, 3: 1.0}],
        ),
        (
            domains[0],
            "FREE",
            "FREE",
            "EXHAUSTIVE",
            surrogate_data_models.BotorchSurrogates(
                surrogates=[
                    surrogate_data_models.SingleTaskGPSurrogate(
                        inputs=domains[0].inputs,
                        outputs=domains[0].outputs,
                    ),
                ],
            ),
            [{1: 1.0}, {1: 2.0}],
        ),
        (
            domains[0],
            "EXHAUSTIVE",
            "FREE",
            "EXHAUSTIVE",
            surrogate_data_models.BotorchSurrogates(
                surrogates=[
                    surrogate_data_models.SingleTaskGPSurrogate(
                        inputs=domains[0].inputs,
                        outputs=domains[0].outputs,
                    ),
                ],
            ),
            [
                {2: 1.0, 3: 2.0, 1: 1.0},
                {2: 3.0, 3: 7.0, 1: 1.0},
                {2: 5.0, 3: 1.0, 1: 1.0},
                {2: 1.0, 3: 2.0, 1: 2.0},
                {2: 3.0, 3: 7.0, 1: 2.0},
                {2: 5.0, 3: 1.0, 1: 2.0},
            ],
        ),
        (
            domains[0],
            "FREE",
            "FREE",
            "FREE",
            surrogate_data_models.BotorchSurrogates(
                surrogates=[
                    surrogate_data_models.SingleTaskGPSurrogate(
                        inputs=domains[0].inputs,
                        outputs=domains[0].outputs,
                        input_preprocessing_specs={
                            "if5": CategoricalEncodingEnum.ONE_HOT,
                        },
                    ),
                ],
            ),
            [{}],
        ),
        (
            domains[0],
            "FREE",
            "EXHAUSTIVE",
            "FREE",
            surrogate_data_models.BotorchSurrogates(surrogates=[]),
            [
                {4: 1.0, 5: 0.0, 6: 0.0},
                {4: 0.0, 5: 1.0, 6: 0.0},
                {4: 0.0, 5: 0.0, 6: 1.0},
            ],
        ),
        (
            domains[3],
            "EXHAUSTIVE",
            "EXHAUSTIVE",
            "EXHAUSTIVE",
            surrogate_data_models.BotorchSurrogates(surrogates=[]),
            [
                {1: 3.0},
            ],
        ),
    ],
)
def test_base_get_categorical_combinations(
    domain,
    descriptor_method,
    categorical_method,
    discrete_method,
    surrogate_specs,
    expected,
):
    data_model = DummyStrategyDataModel(
        domain=domain,
        surrogate_specs=surrogate_specs,
        acquisition_optimizer=data_models.BotorchOptimizer(
            descriptor_method=descriptor_method,
            categorical_method=categorical_method,
            discrete_method=discrete_method,
        ),
    )
    myStrategy = DummyStrategy(data_model=data_model)
    c = unittest.TestCase()
    combo = myStrategy.acqf_optimizer.get_categorical_combinations(
        myStrategy.domain, myStrategy.input_preprocessing_specs
    )
    c.assertCountEqual(combo, expected)


@pytest.mark.parametrize("domain", [(domains[0])])
def test_base_invalid_pair_encoding_method(domain):
    with pytest.raises(ValueError):
        DummyStrategyDataModel(
            domain=domain,
            categorical_encoding="ORDINAL",
            categorical_method="FREE",
        )


@pytest.mark.parametrize(
    "domain, data",
    [
        (
            domains[0],
            generate_experiments(
                domains[0],
                row_count=5,
                tol=1.0,
                force_all_categories=True,
            ),
        ),
        (
            domains[1],
            generate_experiments(
                domains[1],
                row_count=5,
                tol=1.0,
                force_all_categories=True,
            ),
        ),
        (
            domains[2],
            generate_experiments(
                domains[2],
                row_count=5,
                tol=1.0,
                force_all_categories=True,
            ),
        ),
        (
            domains[4],
            generate_experiments(
                domains[4],
                row_count=5,
                tol=1.0,
                force_all_categories=True,
            ),
        ),
    ],
)
def test_base_fit(domain, data):
    data_model = DummyStrategyDataModel(domain=domain)
    myStrategy = DummyStrategy(data_model=data_model)
    myStrategy.set_experiments(data)
    myStrategy.fit()


# TODO: replace this with proper benchmark methods
@pytest.mark.parametrize(
    "domain, data, acquisition_function",
    [
        (
            domains[0],
            generate_experiments(
                domains[0],
                row_count=10,
                tol=1.0,
                force_all_categories=True,
            ),
            specs.acquisition_functions.valid().obj(),
        ),
        (
            domains[1],
            generate_experiments(
                domains[1],
                row_count=10,
                tol=1.0,
                force_all_categories=True,
            ),
            specs.acquisition_functions.valid().obj(),
        ),
        # TODO: this tests randomly fails (All attempts to fit the model have failed.)
        # (
        #     domains[2],
        #     generate_experiments(
        #         domains[2], row_count=10, tol=1.0, force_all_categories=True
        #     ),
        #     specs.acquisition_functions.valid().obj(),
        # ),
        # TODO: this tests randomly fails (All attempts to fit the model have failed.)
        # (
        #     domains[4],
        #     generate_experiments(
        #         domains[4], row_count=10, tol=1.0, force_all_categories=True
        #     ),
        #     specs.acquisition_functions.valid().obj(),
        # ),
    ],
)
def test_base_predict(domain, data, acquisition_function):
    data_model = DummyStrategyDataModel(
        domain=domain,
    )  # , acquisition_function=acquisition_function
    # )
    myStrategy = DummyStrategy(data_model=data_model)
    myStrategy.tell(experiments=data)
    predictions = myStrategy.predict(data)
    assert len(predictions.columns.tolist()) == 3 * len(domain.outputs.get_keys(Output))
    assert data.index[-1] == predictions.index[-1]


@pytest.mark.parametrize(
    "categorical_method, descriptor_method, discrete_method",
    list(
        itertools.product(
            [CategoricalMethodEnum.FREE, CategoricalMethodEnum.EXHAUSTIVE],
            [CategoricalMethodEnum.FREE, CategoricalMethodEnum.EXHAUSTIVE],
            [CategoricalMethodEnum.FREE, CategoricalMethodEnum.EXHAUSTIVE],
        ),
    ),
)
def test_base_setup_ask_fixed_features(
    categorical_method,
    descriptor_method,
    discrete_method,
):
    # test for fixed features list
    data_model = DummyStrategyDataModel(
        domain=domains[0],
        # acquisition_function=specs.acquisition_functions.valid().obj(),
        acquisition_optimizer=data_models.BotorchOptimizer(
            categorical_method=categorical_method,
            descriptor_method=descriptor_method,
            discrete_method=discrete_method,
        ),
        surrogate_specs=surrogate_data_models.BotorchSurrogates(
            surrogates=[
                surrogate_data_models.SingleTaskGPSurrogate(
                    inputs=domains[0].inputs,
                    outputs=domains[0].outputs,
                    # input_preprocessing_specs={"if5": CategoricalEncodingEnum.ONE_HOT},
                ),
            ],
        ),
    )
    myStrategy = DummyStrategy(data_model=data_model)
    myStrategy._experiments = domains[0].inputs.sample(3)
    (
        bounds,
        local_bounds,
        ic_generator,
        ic_gen_kwargs,
        nchooseks,
        fixed_features,
        fixed_features_list,
    ) = myStrategy.acqf_optimizer._setup_ask(
        myStrategy.domain, myStrategy.input_preprocessing_specs, myStrategy.experiments
    )
    if any(
        enc == CategoricalMethodEnum.EXHAUSTIVE
        for enc in [
            categorical_method,
            descriptor_method,
            discrete_method,
        ]
    ):
        assert fixed_features is None
        assert fixed_features_list is not None
    else:
        assert fixed_features == {}
        assert fixed_features_list is None

    data_model = DummyStrategyDataModel(
        domain=domains[3],
        # acquisition_function=specs.acquisition_functions.valid().obj(),
        acquisition_optimizer=data_models.BotorchOptimizer(
            categorical_method=categorical_method,
            descriptor_method=descriptor_method,
            discrete_method=discrete_method,
        ),
    )
    myStrategy = DummyStrategy(data_model=data_model)
    myStrategy._experiments = domains[3].inputs.sample(3)
    (
        bounds,
        local_bounds,
        ic_generator,
        ic_gen_kwargs,
        nchooseks,
        fixed_features,
        fixed_features_list,
    ) = myStrategy.acqf_optimizer._setup_ask(
        myStrategy.domain, myStrategy.input_preprocessing_specs, myStrategy.experiments
    )
    assert fixed_features == {1: 3.0}
    assert fixed_features_list is None


def test_base_setup_ask():
    # test for no nchooseks
    benchmark = Hartmann()
    data_model = DummyStrategyDataModel(
        domain=benchmark.domain,
        # acquisition_function=specs.acquisition_functions.valid().obj(),
    )
    myStrategy = DummyStrategy(data_model=data_model)
    myStrategy._experiments = benchmark.f(
        benchmark.domain.inputs.sample(3),
        return_complete=True,
    )
    (
        bounds,
        local_bounds,
        ic_generator,
        ic_gen_kwargs,
        nchooseks,
        fixed_features,
        fixed_features_list,
    ) = myStrategy.acqf_optimizer._setup_ask(
        myStrategy.domain, myStrategy.input_preprocessing_specs, myStrategy.experiments
    )
    assert torch.allclose(
        bounds,
        torch.tensor([[0 for _ in range(6)], [1 for _ in range(6)]]).to(**tkwargs),
    )
    assert torch.allclose(
        local_bounds,
        torch.tensor([[0 for _ in range(6)], [1 for _ in range(6)]]).to(**tkwargs),
    )
    assert ic_generator is None
    assert ic_gen_kwargs == {}
    assert nchooseks is None
    assert fixed_features == {}
    assert fixed_features_list is None
    # test for nchooseks
    benchmark = Hartmann(dim=6, allowed_k=3)
    data_model = DummyStrategyDataModel(
        domain=benchmark.domain,
        # acquisition_function=specs.acquisition_functions.valid().obj(),
    )
    myStrategy = DummyStrategy(data_model=data_model)
    myStrategy._experiments = benchmark.f(
        benchmark.domain.inputs.sample(3),
        return_complete=True,
    )
    (
        bounds,
        local_bounds,
        ic_generator,
        ic_gen_kwargs,
        nchooseks,
        fixed_features,
        fixed_features_list,
    ) = myStrategy.acqf_optimizer._setup_ask(
        myStrategy.domain, myStrategy.input_preprocessing_specs, myStrategy.experiments
    )
    assert torch.allclose(
        bounds,
        torch.tensor([[0 for _ in range(6)], [1 for _ in range(6)]]).to(**tkwargs),
    )
    assert ic_generator == gen_batch_initial_conditions
    assert list(ic_gen_kwargs.keys()) == ["generator"]
    assert len(nchooseks) == len(get_nchoosek_constraints(domain=benchmark.domain))
    assert fixed_features == {}
    assert fixed_features_list is None
    # test for nchooseks with product constraints
    benchmark = Hartmann(dim=6, allowed_k=3)
    benchmark.domain.constraints.constraints.append(
        ProductInequalityConstraint(
            features=["x_1", "x_2", "x_3"],
            exponents=[1, 1, 1],
            rhs=50,
        ),
    )
    data_model = DummyStrategyDataModel(
        domain=benchmark.domain,
        # acquisition_function=specs.acquisition_functions.valid().obj(),
    )
    myStrategy = DummyStrategy(data_model=data_model)
    myStrategy._experiments = benchmark.f(
        benchmark.domain.inputs.sample(3),
        return_complete=True,
    )
    (
        bounds,
        local_bounds,
        ic_generator,
        ic_gen_kwargs,
        nonlinears,
        fixed_features,
        fixed_features_list,
    ) = myStrategy.acqf_optimizer._setup_ask(
        myStrategy.domain, myStrategy.input_preprocessing_specs, myStrategy.experiments
    )
    assert torch.allclose(
        bounds,
        torch.tensor([[0 for _ in range(6)], [1 for _ in range(6)]]).to(**tkwargs),
    )
    assert ic_generator == gen_batch_initial_conditions
    assert list(ic_gen_kwargs.keys()) == ["generator"]
    assert len(nonlinears) != len(get_nchoosek_constraints(domain=benchmark.domain))
    assert len(nonlinears) == len(get_nonlinear_constraints(domain=benchmark.domain))
    assert fixed_features == {}
    assert fixed_features_list is None
