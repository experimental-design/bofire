import pytest

from bofire.domain.features import ContinuousInput, ContinuousOutput
from bofire.strategies.botorch.base import BotorchBasicBoStrategy, ModelSpec
from bofire.strategies.botorch.sobo import BoTorchSoboStrategy
from bofire.strategies.strategy import Strategy
from tests.bofire.domain.test_features import (
    VALID_CONTINUOUS_INPUT_FEATURE_SPEC,
    VALID_CONTINUOUS_OUTPUT_FEATURE_SPEC,
)
from tests.bofire.strategies.botorch.test_model_spec import VALID_MODEL_SPEC_SPEC
from tests.bofire.strategies.botorch.test_sobo import VALID_BOTORCH_SOBO_STRATEGY_SPEC

if1 = ContinuousInput(
    **{
        **VALID_CONTINUOUS_INPUT_FEATURE_SPEC,
        "key": "if1",
    }
)
if2 = ContinuousInput(
    **{
        **VALID_CONTINUOUS_INPUT_FEATURE_SPEC,
        "key": "if2",
    }
)
of1 = ContinuousOutput(**{**VALID_CONTINUOUS_OUTPUT_FEATURE_SPEC, "key": "of1"})
of2 = ContinuousOutput(**{**VALID_CONTINUOUS_OUTPUT_FEATURE_SPEC, "key": "of2"})


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
                    "model_specs": [
                        ModelSpec(
                            **{
                                **VALID_MODEL_SPEC_SPEC,
                                "output_feature": "of1",
                                "input_features": ["if1"],
                            }
                        )
                    ],
                },
            ),
            2,
        ),
    ],
)
def test_generate_model_specs(strategy: Strategy, expected_count: int):
    model_specs = BotorchBasicBoStrategy._generate_model_specs(
        domain=strategy.domain, model_specs=strategy.model_specs
    )
    assert len(model_specs) == expected_count


@pytest.mark.parametrize(
    "strategy, specs",
    [
        (
            BoTorchSoboStrategy(
                **VALID_BOTORCH_SOBO_STRATEGY_SPEC,
            ),
            [
                ModelSpec(
                    **{
                        **VALID_MODEL_SPEC_SPEC,
                        "output_feature": "unknown",
                        "input_features": ["if1"],
                    }
                )
            ],
        ),
        (
            BoTorchSoboStrategy(
                **VALID_BOTORCH_SOBO_STRATEGY_SPEC,
            ),
            [
                ModelSpec(
                    **{
                        **VALID_MODEL_SPEC_SPEC,
                        "output_feature": "of1",
                        "input_features": ["unknown"],
                    }
                )
            ],
        ),
    ],
)
def test_generate_model_specs_invalid(strategy: Strategy, specs: ModelSpec):
    with pytest.raises(KeyError):
        BotorchBasicBoStrategy._generate_model_specs(strategy.domain, specs)


"""@pytest.mark.parametrize(
    "strategy, specs",
    [
        BoTorchSoboStrategy(
            domain=Domain(
                input_features=[if1, if2],
                output_features=[of1, of2],
                constraints=[],
            ),
            **VALID_BOTORCH_SOBO_STRATEGY_SPEC,
        ),
        [
            ModelSpec(
                **{
                    **VALID_MODEL_SPEC_SPEC,
                    "output_feature": "of1",
                    "input_features": ["if2"],
                }
            )
        ],
    ],
)
def test_generate_valid_model_specs_not_overwrite(strategy: Strategy, specs: ModelSpec):

    model_specs = Strategy._generate_model_specs(strategy.domain, specs)
    assert len(model_specs) == 2
    model_specs = {model_spec.output_feature: model_spec for model_spec in model_specs}
    assert model_specs["of1"].input_features == ["if2"]
    assert model_specs["of2"].input_features == ["if1", "if2"]"""
