import pytest
from pydantic.error_wrappers import ValidationError

from bofire.domain.constraints import (
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
)
from bofire.domain.domain import Domain
from bofire.domain.features import ContinuousInputFeature, ContinuousOutputFeature
from bofire.strategies.strategy import Strategy
from tests.bofire.domain.test_constraints import (
    VALID_LINEAR_CONSTRAINT_SPEC,
    VALID_NCHOOSEKE_CONSTRAINT_SPEC,
)
from tests.bofire.domain.test_features import (
    VALID_CONTINUOUS_INPUT_FEATURE_SPEC,
    VALID_CONTINUOUS_OUTPUT_FEATURE_SPEC,
)
from tests.bofire.strategies.dummy import DummyStrategy

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
c3 = NChooseKConstraint(**{
    **VALID_NCHOOSEKE_CONSTRAINT_SPEC,
    "features": ["if1","if2"],
})


@pytest.mark.parametrize("domain", [
    (
        Domain(
            input_features=[if1, if2],
            output_features=[of1],
            constraints=constraints,
        ),
    )
    for constraints in [[c1], [c2], [c1, c2]]
])
def test_strategy_constructor(
    domain: Domain,
):
    strategy = DummyStrategy(domain)


@pytest.mark.parametrize("domain", [
    (
        Domain(
            input_features=[if1, if2],
            output_features=[of1],
            constraints=constraints,
        ),
    )
    for constraints in [[c3], [c1, c3], [c2, c3], [c1, c2, c3]]
])
def test_strategy_init_domain_invalid_constraints(

    domain: Domain,
):
    with pytest.raises(ValidationError):
        strategy = DummyStrategy(domain)
