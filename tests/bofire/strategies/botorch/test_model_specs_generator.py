import pytest

from bofire.domain.feature import ContinuousInput, ContinuousOutput
from bofire.domain.features import InputFeatures, OutputFeatures
from bofire.models.gps import SingleTaskGPModel
from bofire.models.torch_models import BotorchModels
from bofire.strategies.botorch.base import BotorchBasicBoStrategy
from bofire.strategies.botorch.sobo import BoTorchSoboStrategy
from bofire.strategies.strategy import Strategy
from tests.bofire.strategies.botorch.test_sobo import VALID_BOTORCH_SOBO_STRATEGY_SPEC
from tests.bofire.strategies.specs import (
    VALID_CONTINUOUS_INPUT_FEATURE_SPEC,
    VALID_CONTINUOUS_OUTPUT_FEATURE_SPEC,
)

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
            BoTorchSoboStrategy(
                **VALID_BOTORCH_SOBO_STRATEGY_SPEC,
            ),
            2,
        ),
        (
            BoTorchSoboStrategy(
                **{
                    **VALID_BOTORCH_SOBO_STRATEGY_SPEC,
                    "model_specs": BotorchModels(
                        models=[
                            SingleTaskGPModel(
                                input_features=VALID_BOTORCH_SOBO_STRATEGY_SPEC[
                                    "domain"
                                ].input_features,
                                output_features=OutputFeatures(
                                    features=[
                                        VALID_BOTORCH_SOBO_STRATEGY_SPEC[
                                            "domain"
                                        ].output_features.get_by_key("of1")
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
def test_generate_model_specs(strategy: Strategy, expected_count: int):
    model_specs = BotorchBasicBoStrategy._generate_model_specs(
        domain=strategy.domain
    )  # , model_specs=strategy.model_specs
    # )
    assert len(model_specs.models) == expected_count


@pytest.mark.parametrize(
    "strategy, specs",
    [
        (
            BoTorchSoboStrategy(
                **VALID_BOTORCH_SOBO_STRATEGY_SPEC,
            ),
            BotorchModels(
                models=[
                    SingleTaskGPModel(
                        input_features=InputFeatures(features=[if1, if2]),
                        output_features=OutputFeatures(
                            features=[
                                VALID_BOTORCH_SOBO_STRATEGY_SPEC[
                                    "domain"
                                ].output_features.get_by_key("of1")
                            ]
                        ),
                    ),
                ]
            ),
        ),
        (
            BoTorchSoboStrategy(
                **VALID_BOTORCH_SOBO_STRATEGY_SPEC,
            ),
            BotorchModels(
                models=[
                    SingleTaskGPModel(
                        input_features=VALID_BOTORCH_SOBO_STRATEGY_SPEC[
                            "domain"
                        ].input_features,
                        output_features=OutputFeatures(features=[of1]),
                    ),
                ]
            ),
        ),
    ],
)
def test_generate_model_specs_invalid(strategy: Strategy, specs: BotorchModels):
    with pytest.raises(ValueError):
        BotorchBasicBoStrategy._generate_model_specs(strategy.domain, specs)
