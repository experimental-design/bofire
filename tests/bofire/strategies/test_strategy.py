import random
from typing import List

import mock
import pandas as pd
import pytest
from _pytest.fixtures import fixture
from pydantic.error_wrappers import ValidationError

from bofire.domain.constraints import (
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
)
from bofire.domain.domain import Domain
from bofire.domain.features import CategoricalInput, ContinuousInput, ContinuousOutput
from bofire.domain.objectives import TargetObjective
from bofire.strategies.strategy import Strategy
from tests.bofire.domain.test_constraints import (
    VALID_LINEAR_CONSTRAINT_SPEC,
    VALID_NCHOOSEKE_CONSTRAINT_SPEC,
)
from tests.bofire.domain.test_domain_validators import (
    generate_candidates,
    generate_experiments,
)
from tests.bofire.domain.test_features import (
    VALID_CATEGORICAL_INPUT_FEATURE_SPEC,
    VALID_CONTINUOUS_INPUT_FEATURE_SPEC,
    VALID_CONTINUOUS_OUTPUT_FEATURE_SPEC,
)
from tests.bofire.strategies.dummy import DummyPredictiveStrategy, DummyStrategy

if1 = ContinuousInput(
    **{**VALID_CONTINUOUS_INPUT_FEATURE_SPEC, "key": "if1", "lower_bound": 0.0}
)
if2 = ContinuousInput(
    **{**VALID_CONTINUOUS_INPUT_FEATURE_SPEC, "key": "if2", "lower_bound": 0.0}
)
if3 = CategoricalInput(
    **{
        **VALID_CATEGORICAL_INPUT_FEATURE_SPEC,
        "key": "if3",
    }
)

of1 = ContinuousOutput(
    **{
        **VALID_CONTINUOUS_OUTPUT_FEATURE_SPEC,
        "key": "of1",
    }
)
of2 = ContinuousOutput(
    **{
        **VALID_CONTINUOUS_OUTPUT_FEATURE_SPEC,
        "key": "of2",
    }
)
of3 = ContinuousOutput(key="of3", objective=None)

of4 = ContinuousOutput(
    key="of4",
    objective=TargetObjective(w=1, target_value=5.0, tolerance=1.0, steepness=0.5),
)

c1 = LinearEqualityConstraint(
    **{
        **VALID_LINEAR_CONSTRAINT_SPEC,
        "features": ["if1", "if2"],
        "coefficients": [1, 1],
    }
)
c2 = LinearInequalityConstraint(
    **{
        **VALID_LINEAR_CONSTRAINT_SPEC,
        "features": ["if1", "if2"],
        "coefficients": [1, 1],
    }
)
c3 = NChooseKConstraint(
    **{
        **VALID_NCHOOSEKE_CONSTRAINT_SPEC,
        "features": ["if1", "if2"],
    }
)


@fixture
def strategy():
    return DummyStrategy()


@pytest.mark.parametrize(
    "domain",
    [
        (
            Domain(
                input_features=[if1, if2],
                output_features=[of1],
                constraints=constraints,
            )
        )
        for constraints in [[c1], [c2], [c1, c2]]
    ],
)
def test_strategy_constructor(
    domain: Domain,
):
    print(domain)
    DummyStrategy(domain=domain)


@pytest.mark.parametrize(
    "domain",
    [
        (
            Domain(
                input_features=[if1, if2],
                output_features=[of1],
                constraints=constraints,
            )
        )
        for constraints in [[c3], [c1, c3], [c2, c3], [c1, c2, c3]]
    ],
)
def test_strategy_init_domain_invalid_constraints(
    domain: Domain,
):
    with pytest.raises(ValidationError):
        DummyStrategy(domain=domain)


@pytest.mark.parametrize(
    "domain",
    [
        (
            Domain(
                input_features=input_features,
                output_features=[of1],
                constraints=[],
            )
        )
        for input_features in [[if3], [if1, if3]]
    ],
)
def test_strategy_init_domain_invalid_input(domain: Domain):
    with pytest.raises(ValidationError):
        DummyStrategy(domain=domain)


@pytest.mark.parametrize(
    "domain",
    [
        (
            Domain(
                input_features=[if1, if2],
                output_features=output_features,
                constraints=[],
            )
        )
        for output_features in [[of1, of4], [of4]]
    ],
)
def test_strategy_init_domain_invalid_objective(domain: Domain):
    with pytest.raises(ValidationError):
        DummyStrategy(domain=domain)


def test_strategy_init_domain_noobjective():
    domain = Domain(
        input_features=[if1, if2],
        output_features=[of3],
        constraints=[],
    )
    with pytest.raises(ValidationError):
        DummyStrategy(domain=domain)


domain = Domain(
    input_features=[if1, if2],
    output_features=[of1, of2],
    constraints=[],
)
e1 = generate_experiments(domain, 1)
e2 = generate_experiments(domain, 2)
e3 = generate_experiments(domain, 3)
e4 = generate_experiments(domain, 4)


@pytest.mark.parametrize(
    "domain, experiments, replace",
    [
        (
            Domain(
                input_features=[if1, if2],
                output_features=[of1, of2],
                constraints=[],
            ),
            experiments,
            replace,
        )
        for experiments in [e1, e2]
        for replace in [True, False]
    ],
)
def test_strategy_tell_initial(
    domain: Domain,
    experiments: pd.DataFrame,
    replace: bool,
):
    """verify that tell correctly stores initial experiments"""
    strategy = DummyStrategy(domain=domain)
    print("mama", strategy.domain.experiments)  # , strategy.domain.experiments.shape)
    # print("mama", experiments.shape)
    strategy.tell(experiments=experiments, replace=replace)
    print("papa", strategy.domain.experiments, strategy.domain.experiments.shape)
    assert strategy.domain.experiments.equals(experiments)


@pytest.mark.parametrize(
    "domain, experimentss",
    [
        (
            Domain(
                input_features=[if1, if2],
                output_features=[of1, of2],
                constraints=[],
            ),
            experimentss,
        )
        for experimentss in [[e1, e2], [e2, e1], [e1, e2, e3, e4]]
    ],
)
def test_strategy_tell_append(
    domain: Domain,
    experimentss: List[pd.DataFrame],
):
    strategy = DummyStrategy(domain=domain)
    for index, experiments in enumerate(experimentss):
        strategy.tell(experiments=experiments, replace=False)
        expected_len = sum([len(e) for e in experimentss[: index + 1]])
        assert len(strategy.domain.experiments) == expected_len


@pytest.mark.parametrize(
    "domain, experimentss",
    [(domain, experimentss) for experimentss in [[e1, e2], [e2, e1], [e1, e2, e3, e4]]],
)
def test_strategy_tell_replace(
    domain: Domain,
    experimentss: List[pd.DataFrame],
):
    strategy = DummyStrategy(domain=domain)
    for experiments in experimentss:
        strategy.tell(experiments=experiments, replace=True)
        expected_len = len(experiments)
        assert len(strategy.domain.experiments) == expected_len


@pytest.mark.parametrize("domain, experiments", [(domain, e) for e in [e3, e4]])
def test_strategy_ask_invalid_candidates(
    domain: Domain,
    experiments: pd.DataFrame,
):
    strategy = DummyStrategy(domain=domain)
    strategy.tell(experiments)

    def test_ask(self: Strategy, candidate_count: int):
        candidates = generate_candidates(self.domain, candidate_count)
        candidates = candidates.drop(random.choice(candidates.columns), axis=1)
        return candidates

    with mock.patch.object(DummyStrategy, "_ask", new=test_ask):
        with pytest.raises(ValueError):
            strategy.ask(candidate_count=1)


@pytest.mark.parametrize("domain, experiments", [(domain, e) for e in [e3, e4]])
def test_strategy_ask_invalid_candidate_count(
    domain: Domain,
    experiments: pd.DataFrame,
):
    strategy = DummyStrategy(domain=domain)
    strategy.tell(experiments)

    def test_ask(self: Strategy, candidate_count: int):
        candidates = generate_candidates(self.domain, candidate_count)[:-1]
        return candidates

    with mock.patch.object(DummyStrategy, "_ask", new=test_ask):
        with pytest.raises(ValueError):
            strategy.ask(candidate_count=4)


@pytest.mark.parametrize("domain, experiments", [(domain, e) for e in [e3, e4]])
def test_strategy_ask_valid(
    domain: Domain,
    experiments: pd.DataFrame,
):
    strategy = DummyStrategy(domain=domain)
    strategy.tell(experiments)

    def test_ask(self: Strategy, candidate_count: int):
        candidates = generate_candidates(self.domain, candidate_count)
        return candidates

    with mock.patch.object(DummyStrategy, "_ask", new=test_ask):
        strategy.ask(candidate_count=1)


@pytest.mark.parametrize(
    "domain, experiments, candidate_pool, candidate_count",
    [
        [domain, e3, generate_candidates(domain, 3), 2],
        [domain, e3, generate_candidates(domain, 5), 3],
    ],
)
def test_strategy_ask_valid_candidate_pool(
    domain, experiments, candidate_pool, candidate_count
):
    strategy = DummyStrategy(domain=domain)
    strategy.tell(experiments)
    strategy.ask(candidate_count=candidate_count, candidate_pool=candidate_pool)


@pytest.mark.parametrize(
    "domain, experiments, candidate_pool, candidate_count",
    [
        [domain, e3, generate_candidates(domain, 3), -1],
        [domain, e3, generate_candidates(domain, 3), 4],
    ],
)
def test_ask_invalid_candidate_count_request_pool(
    domain, experiments, candidate_pool, candidate_count
):
    strategy = DummyStrategy(domain=domain)
    strategy.tell(experiments)
    with pytest.raises((AssertionError, ValueError)):
        strategy.ask(candidate_count=candidate_count, candidate_pool=candidate_pool)


def test_ask_invalid_candidate_count_request():
    strategy = DummyStrategy(domain=domain)
    strategy.tell(e3)
    with pytest.raises(ValueError):
        strategy.ask(-1)


# test for PredictiveStrategy
@pytest.mark.parametrize(
    "domain, experiments",
    [
        (
            Domain(
                input_features=[if1, if2],
                output_features=[of1, of2],
                constraints=[],
            ),
            e,
        )
        for e in [e3, e4]
    ],
)
def test_predictive_strategy_ask_valid(
    domain: Domain,
    experiments: pd.DataFrame,
):
    strategy = DummyPredictiveStrategy(domain=domain)
    strategy.tell(experiments)

    def test_ask(self: Strategy, candidate_count: int):
        candidates = generate_candidates(self.domain, candidate_count)
        return candidates

    with mock.patch.object(DummyPredictiveStrategy, "_ask", new=test_ask):
        strategy.ask(candidate_count=1)


@pytest.mark.parametrize(
    "domain, experiments",
    [
        (
            Domain(
                input_features=[if1, if2],
                output_features=[of1, of2],
                constraints=[],
            ),
            e,
        )
        for e in [e3, e4]
    ],
)
def test_predictive_strategy_predict(domain, experiments):
    strategy = DummyPredictiveStrategy(domain=domain)
    strategy.tell(experiments)
    strategy.predict(generate_candidates(domain=domain))


@pytest.mark.parametrize(
    "domain",
    [
        (
            Domain(
                input_features=[if1, if2],
                output_features=[of1, of2],
                constraints=[],
            )
        )
    ],
)
def test_predictive_strategy_predict_not_fitted(domain):
    strategy = DummyPredictiveStrategy(domain=domain)
    with pytest.raises(ValueError):
        strategy.predict(generate_candidates(domain=domain))
