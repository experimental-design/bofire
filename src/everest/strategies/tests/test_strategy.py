import random
from typing import List

import mock
import pandas as pd
import pytest
from _pytest.fixtures import fixture
from everest.domain.constraints import LinearEqualityConstraint
from everest.domain.domain import Domain
from everest.domain.features import (ContinuousInputFeature,
                                     ContinuousOutputFeature)
from everest.domain.tests.test_constraints import VALID_LINEAR_CONSTRAINT_SPEC
from everest.domain.tests.test_domain_validators import (generate_candidates,
                                                         generate_experiments)
from everest.domain.tests.test_features import (
    VALID_CONTINUOUS_INPUT_FEATURE_SPEC, VALID_CONTINUOUS_OUTPUT_FEATURE_SPEC)
from everest.strategies.strategy import Strategy, ModelSpec
from everest.strategies.tests.dummy import DummyStrategy

if1 = ContinuousInputFeature(**{
    **VALID_CONTINUOUS_INPUT_FEATURE_SPEC,
    "key": "if1",
})
if2 = ContinuousInputFeature(**{
    **VALID_CONTINUOUS_INPUT_FEATURE_SPEC,
    "key": "if2",
})
of1 = ContinuousOutputFeature(**{
    **VALID_CONTINUOUS_OUTPUT_FEATURE_SPEC,
    "key": "of1",
})
of2 = ContinuousOutputFeature(**{
    **VALID_CONTINUOUS_OUTPUT_FEATURE_SPEC,
    "key": "of2",
})
domain = Domain(
    input_features=[if1, if2],
    output_features=[of1, of2],
    constraints=[],
)
e1 = generate_experiments(domain, 1)
e2 = generate_experiments(domain, 2)
e3 = generate_experiments(domain, 3)
e4 = generate_experiments(domain, 4)


@fixture
def strategy():
    return DummyStrategy()


@pytest.mark.parametrize("domain, experiments, replace", [
    (domain, experiments, replace)
    for experiments in [e1, e2]
    for replace in [True, False]
])
def test_strategy_tell_initial(
    strategy: Strategy,
    domain: Domain,
    experiments: pd.DataFrame,
    replace: bool,
):
    """verify that tell correctly stores initial experiments"""
    strategy.init_domain(domain)
    print(strategy.experiments)
    strategy.tell(experiments=experiments, replace=replace)
    assert strategy.experiments.equals(experiments)


@pytest.mark.parametrize("domain, experimentss", [
    (domain, experimentss)
    for experimentss in [[e1, e2], [e2, e1], [e1, e2, e3, e4]]
])
def test_strategy_tell_append(
    strategy: Strategy,
    domain: Domain,
    experimentss: List[pd.DataFrame],
):
    strategy.init_domain(domain)
    for index, experiments in enumerate(experimentss):
        strategy.tell(experiments=experiments, replace=False)
        expected_len = sum([len(e) for e in experimentss[:index+1]])
        assert len(strategy.experiments) == expected_len


@pytest.mark.parametrize("domain, experimentss", [
    (domain, experimentss)
    for experimentss in [[e1, e2], [e2, e1], [e1, e2, e3, e4]]
])
def test_strategy_tell_replace(
    strategy: Strategy,
    domain: Domain,
    experimentss: List[pd.DataFrame],
):
    strategy.init_domain(domain)
    for experiments in experimentss:
        strategy.tell(experiments=experiments, replace=True)
        expected_len = len(experiments)
        assert len(strategy.experiments) == expected_len


@pytest.mark.parametrize("experiments", [e1, e2, e3, e4])
def test_strategy_tell_without_domain_init(
    strategy: Strategy,
    experiments: pd.DataFrame,
):
    with pytest.raises(ValueError):
        strategy.tell(experiments)


@pytest.mark.parametrize("domain, experiments", [
    (domain, e)
    for e in [e1, e2]
])
def test_strategy_ask_insufficient_experiments(
    strategy: Strategy,
    domain: Domain,
    experiments: pd.DataFrame,
):
    strategy.init_domain(domain)
    strategy.tell(experiments)
    with pytest.raises(ValueError):
        strategy.ask(candidate_count=1, allow_insufficient_experiments=False)


@pytest.mark.parametrize("domain, experiments", [
    (domain, e)
    for e in [e3, e4]
])
def test_strategy_ask_invalid_candidates(
    strategy: Strategy,
    domain: Domain,
    experiments: pd.DataFrame,
):
    strategy.init_domain(domain)
    strategy.tell(experiments)

    def test_ask(
        self: Strategy,
        candidate_count: int,
        allow_insufficient_experiments: bool = False,
    ):
        candidates = generate_candidates(self.domain, candidate_count)
        candidates = candidates.drop(random.choice(
            candidates.columns
        ), axis=1)
        configs = [{} for _ in range(len(candidates))]
        return candidates, configs
    with mock.patch.object(DummyStrategy, '_ask', new=test_ask):
        with pytest.raises(ValueError):
            strategy.ask(candidate_count=1, allow_insufficient_experiments=False)


@pytest.mark.parametrize("domain, experiments", [
    (domain, e)
    for e in [e3, e4]
])
def test_strategy_ask_different_lengths(
    strategy: Strategy,
    domain: Domain,
    experiments: pd.DataFrame,
):
    strategy.init_domain(domain)
    strategy.tell(experiments)

    def test_ask(
        self: Strategy,
        candidate_count: int,
        allow_insufficient_experiments: bool = False,
    ):
        candidates = generate_candidates(self.domain, candidate_count)
        configs = [{} for _ in range(len(candidates) + 1)]
        return candidates, configs
    with mock.patch.object(DummyStrategy, '_ask', new=test_ask):
        with pytest.raises(ValueError):
            strategy.ask(candidate_count=1, allow_insufficient_experiments=False)


@pytest.mark.parametrize("domain, experiments", [
    (domain, e)
    for e in [e3, e4]
])
def test_strategy_ask_invalid_cnandidate_count(
    strategy: Strategy,
    domain: Domain,
    experiments: pd.DataFrame,
):
    strategy.init_domain(domain)
    strategy.tell(experiments)

    def test_ask(
        self: Strategy,
        candidate_count: int,
        allow_insufficient_experiments: bool = False,
    ):
        candidates = generate_candidates(self.domain, candidate_count)[:-1]
        configs = [{} for _ in range(len(candidates))]
        return candidates, configs
    with mock.patch.object(DummyStrategy, '_ask', new=test_ask):
        with pytest.raises(ValueError):
            strategy.ask(candidate_count=1, allow_insufficient_experiments=False)


@pytest.mark.parametrize("domain, experiments", [
    (domain, e)
    for e in [e3, e4]
])
def test_strategy_ask_valid(
    strategy: Strategy,
    domain: Domain,
    experiments: pd.DataFrame,
):
    strategy.init_domain(domain)
    strategy.tell(experiments)

    def test_ask(
        self: Strategy,
        candidate_count: int,
        allow_insufficient_experiments: bool = False,
    ):
        candidates = generate_candidates(self.domain, candidate_count)
        configs = [{} for _ in range(len(candidates))]
        return candidates, configs
    with mock.patch.object(DummyStrategy, '_ask', new=test_ask):
        strategy.ask(candidate_count=1, allow_insufficient_experiments=False)


@pytest.mark.parametrize("domain, experiments, candidate_count", [
    (domain, e, candidate_count)
    for e in [e1, e2]
    for candidate_count in range(1, 5)
])
def test_strategy_ask_using_random(
    strategy: Strategy,
    domain: Domain,
    experiments: pd.DataFrame,
    candidate_count: int,
):
    strategy.init_domain(domain)
    strategy.tell(experiments)
    candidates, configs = strategy.ask(
        candidate_count=candidate_count,
        allow_insufficient_experiments=True,
    )
    assert len(candidates) == candidate_count


def test_strategy_is_reduceable():
    st = DummyStrategy.from_domain(domain)
    assert st.is_reduceable(domain) == True
    st = DummyStrategy.from_domain(
        domain = domain,
        model_specs =  [
            ModelSpec(
                output_feature = "of1",
                input_features = ["if1", "if2"],
                kernel = "RBF",
                ard = True,
                scaler = "NORMALIZE"
            ),
            ModelSpec(
                output_feature = "of2",
                input_features = ["if1", "if2"],
                kernel = "RBF",
                ard = True,
                scaler = "NORMALIZE"
            ),
        ])
    assert st.is_reduceable(domain) == True
    st = DummyStrategy.from_domain(
        domain = domain,
        model_specs =  [
            ModelSpec(
                output_feature = "of1",
                input_features = ["if1", "if2"],
                kernel = "RBF",
                ard = True,
                scaler = "NORMALIZE"
            ),
            ModelSpec(
                output_feature = "of2",
                input_features = ["if1"],
                kernel = "RBF",
                ard = True,
                scaler = "NORMALIZE"
            ),
        ])
    assert st.is_reduceable(domain) == False
