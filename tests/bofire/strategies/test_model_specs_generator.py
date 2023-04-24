import pytest

import bofire.data_models.strategies.api as data_models
import bofire.data_models.surrogates.api as surrogate_data_models
from bofire.data_models.domain.api import Inputs, Outputs
from bofire.data_models.features.api import ContinuousInput, ContinuousOutput
from bofire.strategies.strategy import Strategy
from tests.bofire.strategies.specs import (
    VALID_CONTINUOUS_INPUT_FEATURE_SPEC,
    VALID_CONTINUOUS_OUTPUT_FEATURE_SPEC,
)
from tests.bofire.strategies.test_qehvi import VALID_BOTORCH_QEHVI_STRATEGY_SPEC

if1 = ContinuousInput(
    **{
        **VALID_CONTINUOUS_INPUT_FEATURE_SPEC,
        "key": "if11",
    }
)
if2 = ContinuousInput(
    **{
        **VALID_CONTINUOUS_INPUT_FEATURE_SPEC,
        "key": "if22",
    }
)
of1 = ContinuousOutput(**{**VALID_CONTINUOUS_OUTPUT_FEATURE_SPEC, "key": "of11"})
of2 = ContinuousOutput(**{**VALID_CONTINUOUS_OUTPUT_FEATURE_SPEC, "key": "of22"})


@pytest.mark.parametrize(
    "strategy, expected_count",
    [
        (
            data_models.QnehviStrategy(
                **VALID_BOTORCH_QEHVI_STRATEGY_SPEC,
            ),
            2,
        ),
        (
            data_models.QnehviStrategy(
                **{
                    **VALID_BOTORCH_QEHVI_STRATEGY_SPEC,
                    "surrogate_specs": surrogate_data_models.BotorchSurrogates(
                        surrogates=[
                            surrogate_data_models.SingleTaskGPSurrogate(
                                inputs=VALID_BOTORCH_QEHVI_STRATEGY_SPEC[
                                    "domain"
                                ].inputs,
                                outputs=Outputs(
                                    features=[
                                        VALID_BOTORCH_QEHVI_STRATEGY_SPEC[
                                            "domain"
                                        ].outputs.get_by_key("of1")
                                    ]
                                ),
                            ),
                        ]
                    ),
                },
            ),
            2,
        ),
    ],
)
def test_generate_surrogate_specs(strategy: Strategy, expected_count: int):
    surrogate_specs = data_models.BotorchStrategy._generate_surrogate_specs(
        domain=strategy.domain
    )  # , surrogate_specs=strategy.surrogate_specs
    # )
    assert len(surrogate_specs.surrogates) == expected_count


@pytest.mark.parametrize(
    "strategy, specs",
    [
        (
            data_models.QnehviStrategy(
                **VALID_BOTORCH_QEHVI_STRATEGY_SPEC,
            ),
            surrogate_data_models.BotorchSurrogates(
                surrogates=[
                    surrogate_data_models.SingleTaskGPSurrogate(
                        inputs=Inputs(features=[if1, if2]),
                        outputs=Outputs(
                            features=[
                                VALID_BOTORCH_QEHVI_STRATEGY_SPEC[
                                    "domain"
                                ].outputs.get_by_key("of1")
                            ]
                        ),
                    ),
                ]
            ),
        ),
        (
            data_models.QnehviStrategy(
                **VALID_BOTORCH_QEHVI_STRATEGY_SPEC,
            ),
            surrogate_data_models.BotorchSurrogates(
                surrogates=[
                    surrogate_data_models.SingleTaskGPSurrogate(
                        inputs=VALID_BOTORCH_QEHVI_STRATEGY_SPEC["domain"].inputs,
                        outputs=Outputs(features=[of1]),
                    ),
                ]
            ),
        ),
    ],
)
def test_generate_surrogate_specs_invalid(
    strategy: data_models.Strategy, specs: surrogate_data_models.BotorchSurrogates
):
    with pytest.raises(ValueError):
        data_models.BotorchStrategy._generate_surrogate_specs(strategy.domain, specs)
