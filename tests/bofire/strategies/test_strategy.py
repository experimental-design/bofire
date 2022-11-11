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
from bofire.domain.features import ContinuousInputFeature, ContinuousOutputFeature
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
    VALID_CONTINUOUS_INPUT_FEATURE_SPEC,
    VALID_CONTINUOUS_OUTPUT_FEATURE_SPEC,
)
from tests.bofire.strategies.dummy import DummyStrategy

if1 = ContinuousInputFeature(
    **{**VALID_CONTINUOUS_INPUT_FEATURE_SPEC, "key": "if1", "lower_bound": 0.0}
)
if2 = ContinuousInputFeature(
    **{**VALID_CONTINUOUS_INPUT_FEATURE_SPEC, "key": "if2", "lower_bound": 0.0}
)
of1 = ContinuousOutputFeature(
    **{
        **VALID_CONTINUOUS_OUTPUT_FEATURE_SPEC,
        "key": "of1",
    }
)
of2 = ContinuousOutputFeature(
    **{
        **VALID_CONTINUOUS_OUTPUT_FEATURE_SPEC,
        "key": "of2",
    }
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
    DummyStrategy(domain)


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
        DummyStrategy(domain)


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
        (domain, experiments, replace)
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
    strategy = DummyStrategy(domain)
    print(strategy.domain.experiments)
    strategy.tell(experiments=experiments, replace=replace)
    assert strategy.domain.experiments.equals(experiments)


@pytest.mark.parametrize(
    "domain, experimentss",
    [(domain, experimentss) for experimentss in [[e1, e2], [e2, e1], [e1, e2, e3, e4]]],
)
def test_strategy_tell_append(
    domain: Domain,
    experimentss: List[pd.DataFrame],
):
    strategy = DummyStrategy(domain)
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
    strategy = DummyStrategy(domain)
    for experiments in experimentss:
        strategy.tell(experiments=experiments, replace=True)
        expected_len = len(experiments)
        assert len(strategy.domain.experiments) == expected_len


@pytest.mark.parametrize("domain, experiments", [(domain, e) for e in [e3, e4]])
def test_strategy_ask_invalid_candidates(
    domain: Domain,
    experiments: pd.DataFrame,
):
    strategy = DummyStrategy(domain)
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
    strategy = DummyStrategy(domain)
    strategy.tell(experiments)

    def test_ask(self: Strategy, candidate_count: int):
        candidates = generate_candidates(self.domain, candidate_count)[:-1]
        return candidates

    with mock.patch.object(DummyStrategy, "_ask", new=test_ask):
        with pytest.raises(ValueError):
            strategy.ask(candidate_count=1)


@pytest.mark.parametrize("domain, experiments", [(domain, e) for e in [e3, e4]])
def test_strategy_ask_valid(
    domain: Domain,
    experiments: pd.DataFrame,
):
    strategy = DummyStrategy(domain)
    strategy.tell(experiments)

    def test_ask(self: Strategy, candidate_count: int):
        candidates = generate_candidates(self.domain, candidate_count)
        return candidates

    with mock.patch.object(DummyStrategy, "_ask", new=test_ask):
        strategy.ask(candidate_count=1)


def test_ask_invalid_candidate_count_request():
    strategy = DummyStrategy(domain)
    strategy.tell(e3)
    with pytest.raises(ValidationError):
        strategy.ask(-1)
