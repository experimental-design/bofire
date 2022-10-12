import pytest
from everest.domain.constraints import (ConcurrencyConstraint,
                                        LinearEqualityConstraint,
                                        LinearInequalityConstraint)
from everest.domain.domain import Domain
from everest.domain.features import (ContinuousInputFeature,
                                     ContinuousOutputFeature)
from everest.domain.tests.test_constraints import (
    VALID_CONCURRENCY_CONSTRAINT_SPEC, VALID_LINEAR_CONSTRAINT_SPEC)
from everest.domain.tests.test_features import (
    VALID_CONTINUOUS_INPUT_FEATURE_SPEC, VALID_CONTINUOUS_OUTPUT_FEATURE_SPEC)
from everest.strategies.strategy import Strategy
from everest.strategies.tests.dummy import DummyStrategy
from pydantic.error_wrappers import ValidationError

if1 = ContinuousInputFeature(
    lower_bound = 0.0,
    upper_bound = 5.3,
    key= "if1",
)
if2 = ContinuousInputFeature(
    lower_bound = 0.0,
    upper_bound = 5.3,
    key = "if2",
)
of1 = ContinuousOutputFeature(**{
    **VALID_CONTINUOUS_OUTPUT_FEATURE_SPEC,
    "key": "of1",
})
of2 = ContinuousOutputFeature(**{
    **VALID_CONTINUOUS_OUTPUT_FEATURE_SPEC,
    "key": "of2",
})

c1 = LinearEqualityConstraint(**{
    **VALID_LINEAR_CONSTRAINT_SPEC,
    "features": ["if1", "if2"],
    "coefficients": [1, 1],
})
c2 = LinearInequalityConstraint(**{
    **VALID_LINEAR_CONSTRAINT_SPEC,
    "features": ["if1", "if2"],
    "coefficients": [1,1],
})
c3 = ConcurrencyConstraint(**{
    **VALID_CONCURRENCY_CONSTRAINT_SPEC,
    "features": ["if1","if2"],
})


@pytest.mark.parametrize("strategy, domain", [
    (
        DummyStrategy(),
        Domain(
            input_features=[if1, if2],
            output_features=[of1],
            constraints=constraints,
        ),
    )
    for constraints in [[c1], [c2], [c1, c2]]
])
def test_strategy_init_domain(
    strategy: Strategy,
    domain: Domain,
):
    strategy.init_domain(domain)


@pytest.mark.parametrize("strategy, domain", [
    (
        DummyStrategy(),
        Domain(
            input_features=input_features,
            output_features=output_features,
            constraints=[],
        ),
    )
    for input_features in [[if1], [if2], [if1, if2]]
    for output_features in [[of1], [of2], [of1, of2]]
])
def test_strategy_init_domain_model_specs(
    strategy: Strategy,
    domain: Domain,
):
    strategy.init_domain(domain)
    assert len(strategy.model_specs) == len(domain.output_features)
    for model_spec in strategy.model_specs:
        assert len(model_spec.input_features) == len(domain.input_features)


@pytest.mark.parametrize("strategy, domain", [
    (
        DummyStrategy(),
        Domain(
            input_features=[if1, if2],
            output_features=[of1],
            constraints=constraints,
        ),
    )
    for constraints in [[c3], [c1, c3], [c2, c3], [c1, c2, c3]]
])
def test_strategy_init_domain_invalid_constraints(
    strategy: Strategy,
    domain: Domain,
):
    with pytest.raises(ValidationError):
        strategy.init_domain(domain)
