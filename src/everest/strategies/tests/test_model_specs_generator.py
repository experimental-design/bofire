import pytest
from everest.domain.constraints import ConcurrencyConstraint
from everest.domain.domain import Domain
from everest.domain.features import (ContinuousInputFeature,
                                     ContinuousOutputFeature)
from everest.domain.tests.test_features import (
    VALID_CONTINUOUS_INPUT_FEATURE_SPEC, VALID_CONTINUOUS_OUTPUT_FEATURE_SPEC)
from everest.strategies.botorch.sobo import BoTorchSoboStrategy
from everest.strategies.botorch.tests.test_sobo import \
    VALID_BOTORCH_SOBO_STRATEGY_SPEC
from everest.strategies.strategy import ModelSpec, Strategy
from everest.strategies.tests.test_model_spec import VALID_MODEL_SPEC_SPEC

if1 = ContinuousInputFeature(
    **{
        **VALID_CONTINUOUS_INPUT_FEATURE_SPEC,
        "key": "if1",
    }
)
if2 = ContinuousInputFeature(
    **{
        **VALID_CONTINUOUS_INPUT_FEATURE_SPEC,
        "key": "if2",
    }
)
of1 = ContinuousOutputFeature(**{**VALID_CONTINUOUS_OUTPUT_FEATURE_SPEC, "key": "of1"})
of2 = ContinuousOutputFeature(**{**VALID_CONTINUOUS_OUTPUT_FEATURE_SPEC, "key": "of2"})


@pytest.mark.parametrize(
    "strategy, domain, expected_count",
    [
        (
            BoTorchSoboStrategy(
                **VALID_BOTORCH_SOBO_STRATEGY_SPEC,
            ),
            Domain(
                input_features=[],
                output_features=[],
                constraints=[],
            ),
            0,
        ),
        (
            BoTorchSoboStrategy(
                **VALID_BOTORCH_SOBO_STRATEGY_SPEC,
            ),
            Domain(
                input_features=[if1],
                output_features=[of1, of2],
                constraints=[],
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
                }
            ),
            Domain(
                input_features=[if1],
                output_features=[of1, of2],
                constraints=[],
            ),
            2,
        ),
    ],
)
def test_generate_model_specs(strategy: Strategy, domain: Domain, expected_count: int):
    model_specs = Strategy._generate_model_specs(domain, strategy.model_specs)
    assert len(model_specs) == expected_count


@pytest.mark.parametrize(
    "strategy, domain",
    [
        (
            BoTorchSoboStrategy(
                **{
                    **VALID_BOTORCH_SOBO_STRATEGY_SPEC,
                    "model_specs": [
                        ModelSpec(
                            **{
                                **VALID_MODEL_SPEC_SPEC,
                                "output_feature": "unknown",
                                "input_features": ["if1"],
                            }
                        )
                    ],
                }
            ),
            Domain(
                input_features=[if1],
                output_features=[of1],
                constraints=[],
            ),
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
                                "input_features": ["unknown"],
                            }
                        )
                    ],
                }
            ),
            Domain(
                input_features=[if1],
                output_features=[of1],
                constraints=[],
            ),
        ),
    ],
)
def test_generate_model_specs_invalid(strategy: Strategy, domain: Domain):
    with pytest.raises(KeyError):
        Strategy._generate_model_specs(domain, strategy.model_specs)


def test_generate_valid_model_specs_not_overwrite():
    strategy = BoTorchSoboStrategy(
        **{
            **VALID_BOTORCH_SOBO_STRATEGY_SPEC,
            "model_specs": [
                ModelSpec(
                    **{
                        **VALID_MODEL_SPEC_SPEC,
                        "output_feature": "of1",
                        "input_features": ["if2"],
                    }
                )
            ],
        }
    )
    domain = Domain(
        input_features=[if1, if2],
        output_features=[of1, of2],
        constraints=[],
    )
    model_specs = Strategy._generate_model_specs(domain, strategy.model_specs)
    assert len(model_specs) == 2
    model_specs = {
        model_spec.output_feature: model_spec
        for model_spec in model_specs
    }
    assert model_specs["of1"].input_features == ["if2"]
    assert model_specs["of2"].input_features == ["if1", "if2"]
