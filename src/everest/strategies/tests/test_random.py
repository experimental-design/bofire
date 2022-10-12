import random

import pytest
import torch
from everest.domain import Domain
from everest.domain.constraints import (LinearEqualityConstraint,
                                        LinearInequalityConstraint)
from everest.domain.features import (CategoricalInputFeature,
                                     ContinuousInputFeature,
                                     ContinuousOutputFeature)
from everest.domain.tests.test_features import (
    VALID_CATEGORICAL_INPUT_FEATURE_SPEC, VALID_CONTINUOUS_INPUT_FEATURE_SPEC,
    VALID_CONTINUOUS_OUTPUT_FEATURE_SPEC,
    VALID_FIXED_CONTINUOUS_INPUT_FEATURE_SPEC)
from everest.strategies.botorch import tkwargs
from everest.strategies.strategy import RandomStrategy
from everest.utils.reduce import reduce_domain

if1 = ContinuousInputFeature(**{
    **VALID_CONTINUOUS_INPUT_FEATURE_SPEC,
    "key": "if1",
})
if2 = ContinuousInputFeature(**{
    **VALID_CONTINUOUS_INPUT_FEATURE_SPEC,
    "key": "if2",
})
if3 = ContinuousInputFeature(**{
    **VALID_CONTINUOUS_INPUT_FEATURE_SPEC,
    "key": "if3",
})
if4 = CategoricalInputFeature(**{
    **VALID_CATEGORICAL_INPUT_FEATURE_SPEC,
    "key": "if4",
})
if5 = ContinuousInputFeature(**{
    **VALID_FIXED_CONTINUOUS_INPUT_FEATURE_SPEC,
    "key": "if5",
})
of1 = ContinuousOutputFeature(**{
    **VALID_CONTINUOUS_OUTPUT_FEATURE_SPEC,
    "key": "of1",
})
of2 = ContinuousOutputFeature(**{
    **VALID_CONTINUOUS_OUTPUT_FEATURE_SPEC,
    "key": "of2",
})

domains = [
    Domain(
        input_features=[if1, if2, if3],
        output_features=[of1, of2],
        constraints=[],
    ),
    Domain(
        input_features=[if1, if2, if3, if5],
        output_features=[of1, of2],
        constraints=[],
    ),
    Domain(
        input_features=[if1, if2, if3, if4],
        output_features=[of1, of2],
        constraints=[],
    )
    # TODO: tests also constraints
]

@pytest.mark.parametrize("domain, candidate_count, use_sobol, reduce", [
    (domain, candidate_count, use_sobol, reduce)
    for domain in domains
    for candidate_count in range(1,5)
    for use_sobol in [True,False]
    for reduce in [True, False]
])
def test_RandomStrategy(
    domain: Domain,
    candidate_count: int,
    use_sobol: bool,
    reduce: bool
):
    strategy = RandomStrategy(use_sobol=use_sobol, reduce=reduce)
    strategy.init_domain(domain)
    candidates, configs = strategy.ask(
        candidate_count=candidate_count,
        allow_insufficient_experiments=True,
    )
    assert len(candidates) == candidate_count

if1 = ContinuousInputFeature(
    lower_bound = 0.,
    upper_bound = 1.,
    key= "if1",
)
if2 = ContinuousInputFeature(
    lower_bound = 0.,
    upper_bound = 1.,
    key= "if2",
)
if3 = ContinuousInputFeature(
    lower_bound = 0.,
    upper_bound = 1.,
    key= "if3",
)
if4 = ContinuousInputFeature(
    lower_bound = 0.1,
    upper_bound = 0.1,
    key= "if4",
)
i5 = CategoricalInputFeature(
    categories = ["a","b","c"],
    key= "if5",
)
if6 = CategoricalInputFeature(
    categories = ["a","b","c"],
    allowed = [False,True,False],
    key= "if6",
)
c1 = LinearEqualityConstraint(
    features = ["if1","if2","if3","if4"],
    coefficients = [1.,1.,1.,1.],
    rhs = 1.
)
c2 = LinearInequalityConstraint(
    features = ["if1","if2"],
    coefficients = [1.,1.],
    rhs = 0.2
)
c3 = LinearInequalityConstraint(
    features = ["if1","if2","if4"],
    coefficients = [1.,1.,0.5],
    rhs = 0.2
)

domains = [
    Domain(
        input_features=[if1,if2,if3],
        output_features = [of1,of2],
        constraints = [c2]
    ),
    Domain(
        input_features=[if1,if2,if3,if4],
        output_features = [of1,of2],
        constraints = [c1,c2]
    ),
    Domain(
        input_features=[if1,if2,if3,if4],
        output_features = [of1,of2],
        constraints = [c1,c2,c3]
    ),
    Domain(
        input_features=[if1,if2,if3,if5],
        output_features = [of1,of2],
        constraints = [c2]
    ),
    Domain(
        input_features=[if1,if2,if3,if4,if5],
        output_features = [of1,of2],
        constraints = [c1,c2]
    ),
    Domain(
        input_features=[if1,if2,if3,if4,if5],
        output_features = [of1,of2],
        constraints = [c1,c2,c3]
    ),
    Domain(
        input_features=[if1,if2,if3,if6],
        output_features = [of1,of2],
        constraints = [c2]
    ),
    Domain(
        input_features=[if1,if2,if3,if4,if6],
        output_features = [of1,of2],
        constraints = [c1,c2]
    ),
    Domain(
        input_features=[if1,if2,if3,if4,if6],
        output_features = [of1,of2],
        constraints = [c1,c2,c3]
    ),
]

@pytest.mark.parametrize("domain, candidate_count, reduce, unit_scaled", [
    (domain, candidate_count, reduce, unit_scaled)
    for domain in domains
    for candidate_count in range(1,5)
    for reduce in [True, False]
    for unit_scaled in [True, False]
])
def test_RandomStrategyConstraints(domain,candidate_count, reduce, unit_scaled):
    strategy = RandomStrategy(use_sobol=False, reduce=reduce, unit_scaled=unit_scaled)
    strategy.init_domain(domain)
    candidates, configs = strategy.ask(
        candidate_count=candidate_count,
        allow_insufficient_experiments=True,
    )
    assert len(candidates) == candidate_count


def test_RandomStrategyReduce():
    # this test is representative for all strategies using the reduced features
    domain = Domain()
    domain.add_feature(ContinuousInputFeature(key = "x1", lower_bound = 0.1, upper_bound = 1.))
    domain.add_feature(ContinuousInputFeature(key = "x2", lower_bound = 0., upper_bound = 0.8))
    domain.add_feature(ContinuousInputFeature(key = "x3", lower_bound = 0.3, upper_bound = 0.9))
    domain.add_feature(ContinuousOutputFeature(key = "y"))
    domain.add_constraint(LinearEqualityConstraint(features = ["x1","x2","x3"], coefficients = [1.,1.,1.], rhs = 1))

    _domain, transform = reduce_domain(domain)

    rec = RandomStrategy.from_domain(domain, reduce = False)
    data, _ = rec.ask(4)
    data.drop(columns = ["y_pred", "y_sd", "y_des"], inplace=True)
    data["y"] = [random.random() for _ in range(4)]
    data["valid_y"] = 1
    _data = transform.drop_data(data)

    rec = RandomStrategy.from_domain(domain, reduce = True)
    rec.tell(data)

    rec = RandomStrategy.from_domain(domain, reduce = True)
    rec.tell(_data)

    with pytest.raises(ValueError):
        rec = RandomStrategy.from_domain(domain, reduce = False)
        rec.tell(_data)


def test_get_linear_constraints():
    domain = Domain(
        input_features = [if1,if2],
        output_features = [of1],
        constraints = []
    )
    constraints = RandomStrategy.get_linear_constraints(domain,LinearEqualityConstraint)
    assert len(constraints) == 0
    constraints = RandomStrategy.get_linear_constraints(domain,LinearInequalityConstraint)
    assert len(constraints) == 0

    domain = Domain(
        input_features=[if1,if2,if3],
        output_features = [of1,of2],
        constraints = [c2]
    )
    constraints = RandomStrategy.get_linear_constraints(domain,LinearEqualityConstraint)
    assert len(constraints) == 0
    constraints = RandomStrategy.get_linear_constraints(domain,LinearInequalityConstraint)
    assert len(constraints) == 1
    assert constraints[0][2] == c2.rhs

    domain = Domain(
        input_features=[if1,if2,if3,if4],
        output_features = [of1,of2],
        constraints = [c1,c2]
    )
    constraints = RandomStrategy.get_linear_constraints(domain,LinearEqualityConstraint)
    assert len(constraints) == 1
    assert constraints[0][2] == c1.rhs - 0.1
    assert len(constraints[0][0]) == len(c1.features)-1
    assert len(constraints[0][1]) == len(c1.coefficients)-1
    constraints = RandomStrategy.get_linear_constraints(domain,LinearInequalityConstraint)
    assert len(constraints) == 1
    assert constraints[0][2] == c2.rhs
    assert len(constraints[0][0]) == len(c2.features)
    assert len(constraints[0][1]) == len(c2.coefficients)


    domain = Domain(
        input_features=[if1,if2,if3,if4,if5],
        output_features = [of1,of2],
        constraints = [c1,c2,c3]
    )
    constraints = RandomStrategy.get_linear_constraints(domain,LinearEqualityConstraint)
    assert len(constraints) == 1
    assert constraints[0][2] == c1.rhs - 0.1
    assert len(constraints[0][0]) == len(c1.features)-1
    assert len(constraints[0][1]) == len(c1.coefficients)-1
    constraints = RandomStrategy.get_linear_constraints(domain,LinearInequalityConstraint)
    assert len(constraints) == 2
    assert constraints[0][2] == c2.rhs
    assert len(constraints[0][0]) == len(c2.features)
    assert len(constraints[0][1]) == len(c2.coefficients)
    assert constraints[1][2] == c3.rhs - 0.5*0.1
    assert len(constraints[1][0]) == len(c3.features)-1
    assert len(constraints[1][1]) == len(c3.coefficients)-1

def test_get_linear_constraints_unit_scaled():
    domain = Domain()
    domain.add_feature(ContinuousInputFeature(key="base_polymer", lower_bound = 0.3, upper_bound = 0.7))
    domain.add_feature(ContinuousInputFeature(key="glas_fibre", lower_bound = 0.1, upper_bound = 0.7))
    domain.add_feature(ContinuousInputFeature(key="additive", lower_bound = 0.1, upper_bound = 0.6))
    domain.add_feature(ContinuousInputFeature(key="temperature", lower_bound = 30., upper_bound  = 700.))
    domain.add_feature(ContinuousOutputFeature(key="e_module"))
    domain.add_constraint(LinearEqualityConstraint(coefficients=[1.,1.,1.], features = ["base_polymer", "glas_fibre", "additive"], rhs = 1.0))

    constraints = RandomStrategy.get_linear_constraints(domain,LinearEqualityConstraint, unit_scaled=True)
    assert len(constraints) == 1
    assert len(constraints[0][0]) == 3
    assert len(constraints[0][1]) == 3
    assert constraints[0][2] == 0.5
    assert torch.allclose(constraints[0][1], torch.tensor([0.4,0.6,0.5]).to(**tkwargs))

