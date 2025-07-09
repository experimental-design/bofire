from typing import List
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from _pytest.fixtures import fixture
from pandas.testing import assert_frame_equal
from pydantic.error_wrappers import ValidationError

from bofire.data_models.constraints.api import (
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
)
from bofire.data_models.domain.api import Domain, Outputs
from bofire.data_models.features.api import (
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
)
from bofire.data_models.objectives.api import TargetObjective
from bofire.data_models.outlier_detection.api import OutlierDetections
from bofire.data_models.outlier_detection.outlier_detection import IterativeTrimming
from bofire.data_models.surrogates.api import SingleTaskGPSurrogate
from bofire.strategies.strategy import Strategy
from tests.bofire.data_models.domain.test_domain_validators import (
    generate_candidates,
    generate_experiments,
)
from tests.bofire.strategies import dummy
from tests.bofire.strategies.specs import (
    VALID_CATEGORICAL_INPUT_FEATURE_SPEC,
    VALID_CONTINUOUS_INPUT_FEATURE_SPEC,
    VALID_CONTINUOUS_OUTPUT_FEATURE_SPEC,
    VALID_LINEAR_EQUALITY_CONSTRAINT_SPEC,
    VALID_LINEAR_INEQUALITY_CONSTRAINT_SPEC,
    VALID_NCHOOSEKE_CONSTRAINT_SPEC,
)


if1 = ContinuousInput(
    **{**VALID_CONTINUOUS_INPUT_FEATURE_SPEC, "key": "if1", "bounds": (0, 5.3)},
)
if2 = ContinuousInput(
    **{**VALID_CONTINUOUS_INPUT_FEATURE_SPEC, "key": "if2", "bounds": (0, 5.3)},
)
if3 = CategoricalInput(
    **{
        **VALID_CATEGORICAL_INPUT_FEATURE_SPEC,
        "key": "if3",
    },
)

of1 = ContinuousOutput(
    **{
        **VALID_CONTINUOUS_OUTPUT_FEATURE_SPEC,
        "key": "of1",
    },
)
of2 = ContinuousOutput(
    **{
        **VALID_CONTINUOUS_OUTPUT_FEATURE_SPEC,
        "key": "of2",
    },
)
of3 = ContinuousOutput(key="of3", objective=None)

of4 = ContinuousOutput(
    key="of4",
    objective=TargetObjective(w=1, target_value=5.0, tolerance=1.0, steepness=0.5),
)

c1 = LinearEqualityConstraint(
    **{
        **VALID_LINEAR_EQUALITY_CONSTRAINT_SPEC,
        "features": ["if1", "if2"],
        "coefficients": [1, 1],
    },
)
c2 = LinearInequalityConstraint(
    **{
        **VALID_LINEAR_INEQUALITY_CONSTRAINT_SPEC,
        "features": ["if1", "if2"],
        "coefficients": [1, 1],
    },
)
c3 = NChooseKConstraint(
    **{
        **VALID_NCHOOSEKE_CONSTRAINT_SPEC,
        "features": ["if1", "if2"],
    },
)


@fixture
def strategy():
    data_model = dummy.DummyStrategyDataModel(
        domain=Domain.from_lists(
            inputs=[if1, if2],
            outputs=[of1, of2],
            constraints=[],
        ),
    )
    return dummy.DummyStrategy(data_model=data_model)


@pytest.mark.parametrize(
    "domain",
    [
        (
            Domain.from_lists(
                inputs=[if1, if2],
                outputs=[of1],
                constraints=constraints,
            )
        )
        for constraints in [[c1], [c2], [c1, c2]]
    ],
)
def test_strategy_constructor(
    domain: Domain,
):
    dummy.DummyStrategyDataModel(domain=domain)


@pytest.mark.parametrize(
    "domain",
    [
        (
            Domain.from_lists(
                inputs=[if1, if2],
                outputs=[of1],
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
        dummy.DummyStrategyDataModel(domain=domain)


@pytest.mark.parametrize(
    "domain",
    [
        (
            Domain.from_lists(
                inputs=inputs,
                outputs=[of1],
                constraints=[],
            )
        )
        for inputs in [[if3], [if1, if3]]
    ],
)
def test_strategy_init_domain_invalid_input(domain: Domain):
    with pytest.raises(ValidationError):
        dummy.DummyStrategyDataModel(domain=domain)


@pytest.mark.parametrize(
    "domain",
    [
        (
            Domain.from_lists(
                inputs=[if1, if2],
                outputs=outputs,
                constraints=[],
            )
        )
        for outputs in [[of1, of4], [of4]]
    ],
)
def test_strategy_init_domain_invalid_objective(domain: Domain):
    with pytest.raises(ValidationError):
        dummy.DummyStrategyDataModel(domain=domain)


def test_strategy_init_domain_noobjective():
    domain = Domain.from_lists(
        inputs=[if1, if2],
        outputs=[of3],
        constraints=[],
    )
    with pytest.raises(ValidationError):
        dummy.DummyStrategyDataModel(domain=domain)


domain = Domain.from_lists(
    inputs=[if1, if2],
    outputs=[of1, of2],
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
            Domain.from_lists(
                inputs=[if1, if2],
                outputs=[of1, of2],
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
    """Verify that tell correctly stores initial experiments"""
    strategy = dummy.DummyStrategy(
        data_model=dummy.DummyStrategyDataModel(domain=domain),
    )
    strategy.tell(experiments=experiments, replace=replace)
    assert strategy.experiments.equals(experiments)


def test_strategy_no_variance():
    domain = Domain.from_lists(
        inputs=[
            ContinuousInput(
                key="a",
                bounds=(0, 1),
            ),
            ContinuousInput(key="b", bounds=(1, 1)),
        ],
        outputs=[of1],
    )
    experiments = domain.inputs.sample(5)
    experiments["of1"] = [1, 2, 3, 4, 5]
    experiments["valid_of1"] = 1
    strategy = dummy.DummyStrategy(
        data_model=dummy.DummyStrategyDataModel(domain=domain),
    )
    strategy.tell(experiments)
    strategy = dummy.DummyPredictiveStrategy(
        data_model=dummy.DummyPredictiveStrategyDataModel(domain=domain),
    )
    with pytest.raises(ValueError):
        strategy.tell(experiments)
    # introduce variance but in an invalid experiment
    experiments.loc[0, "valid_of1"] = 0
    experiments.loc[0, "b"] = 0.7
    with pytest.raises(ValueError):
        strategy.tell(experiments)
    # introduce variance
    experiments["b"] = 0
    strategy.tell(experiments)


def test_strategy_set_experiments():
    strategy = dummy.DummyStrategy(
        data_model=dummy.DummyStrategyDataModel(domain=domain),
    )
    assert strategy.num_experiments == 0
    experiments = generate_experiments(domain, 2)
    strategy.set_experiments(experiments=experiments)
    assert_frame_equal(strategy.experiments, experiments)
    assert_frame_equal(strategy._experiments, experiments)
    assert strategy.num_experiments == 2


def test_strategy_add_experiments():
    strategy = dummy.DummyStrategy(
        data_model=dummy.DummyStrategyDataModel(domain=domain),
    )
    assert strategy.num_experiments == 0
    experiments = generate_experiments(domain, 2)
    strategy.add_experiments(experiments=experiments)
    assert_frame_equal(strategy.experiments, experiments)
    assert strategy.num_experiments == 2
    experiments2 = generate_experiments(domain, 5)
    strategy.add_experiments(experiments=experiments2)
    assert strategy.num_experiments == 7
    assert_frame_equal(
        strategy.experiments,
        pd.concat((experiments, experiments2), ignore_index=True),
    )


def test_strategy_set_candidates():
    strategy = dummy.DummyStrategy(
        data_model=dummy.DummyStrategyDataModel(domain=domain),
    )
    assert strategy.num_candidates == 0
    candidates = generate_candidates(domain, 2)
    strategy.set_candidates(candidates=candidates)
    assert_frame_equal(strategy.candidates, candidates[domain.inputs.get_keys()])
    assert_frame_equal(strategy._candidates, candidates[domain.inputs.get_keys()])
    assert strategy.num_candidates == 2
    strategy.reset_candidates()
    assert strategy.num_candidates == 0


def test_strategy_add_candidates():
    strategy = dummy.DummyStrategy(
        data_model=dummy.DummyStrategyDataModel(domain=domain),
    )
    assert strategy.num_candidates == 0
    candidates = generate_candidates(domain, 2)
    strategy.add_candidates(candidates=candidates)
    assert_frame_equal(strategy.candidates, candidates[domain.inputs.get_keys()])
    assert strategy.num_candidates == 2
    candidates2 = generate_candidates(domain, 5)
    strategy.add_candidates(candidates=candidates2)
    assert strategy.num_candidates == 7
    assert_frame_equal(
        strategy.candidates,
        pd.concat((candidates, candidates2), ignore_index=True)[
            domain.inputs.get_keys()
        ],
    )


@pytest.mark.parametrize(
    "domain, experimentss",
    [
        (
            Domain.from_lists(
                inputs=[if1, if2],
                outputs=[of1, of2],
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
    strategy = dummy.DummyStrategy(
        data_model=dummy.DummyStrategyDataModel(domain=domain),
    )
    for index, experiments in enumerate(experimentss):
        strategy.tell(experiments=experiments, replace=False)
        expected_len = sum([len(e) for e in experimentss[: index + 1]])
        assert len(strategy.experiments) == expected_len


@pytest.mark.parametrize(
    "domain, experimentss",
    [(domain, experimentss) for experimentss in [[e1, e2], [e2, e1], [e1, e2, e3, e4]]],
)
def test_strategy_tell_replace(
    domain: Domain,
    experimentss: List[pd.DataFrame],
):
    strategy = dummy.DummyStrategy(
        data_model=dummy.DummyStrategyDataModel(domain=domain),
    )
    for experiments in experimentss:
        strategy.tell(experiments=experiments, replace=True)
        expected_len = len(experiments)
        assert len(strategy.experiments) == expected_len


@pytest.mark.parametrize(
    "domain",
    [domain],
)
def test_strategy_tell_outliers(
    domain: Domain,
):
    experiments = generate_experiments(domain=domain, row_count=200)
    outlier_detectors = []
    for i, key in enumerate(domain.outputs.get_keys()):
        experiments.loc[:59, key] = experiments.loc[:59, key] + np.random.randn(60) * 1
        outlier_detectors.append(
            IterativeTrimming(
                base_gp=SingleTaskGPSurrogate(
                    inputs=domain.inputs,
                    outputs=Outputs(features=[domain.outputs[i]]),
                ),
            ),
        )
    experiments = domain.validate_experiments(experiments=experiments)
    experiments1 = experiments.copy()
    strategy = dummy.DummyBotorchPredictiveStrategy(
        data_model=dummy.DummyStrategyDataModel(
            domain=domain,
            outlier_detection_specs=OutlierDetections(detectors=outlier_detectors),
        ),
    )
    strategy1 = dummy.DummyBotorchPredictiveStrategy(
        data_model=dummy.DummyStrategyDataModel(
            domain=domain,
        ),
    )
    strategy.tell(experiments=experiments)
    assert_frame_equal(
        experiments1,
        experiments,
    )  # test that experiments don't get changed outside detect_outliers
    strategy1.tell(experiments=experiments)
    assert str(strategy.model.state_dict()) != str(
        strategy1.model.state_dict(),
    )  # test if two fitted surrogates are different


@pytest.mark.parametrize("domain, experiments", [(domain, e) for e in [e3, e4]])
def test_strategy_ask_invalid_candidate_count(
    domain: Domain,
    experiments: pd.DataFrame,
):
    strategy = dummy.DummyStrategy(
        data_model=dummy.DummyStrategyDataModel(domain=domain),
    )
    strategy.tell(experiments)

    def test_ask(self: Strategy, candidate_count: int):
        candidates = generate_candidates(self.domain, candidate_count)[:-1]
        return candidates

    with mock.patch.object(dummy.DummyStrategy, "_ask", new=test_ask):
        with pytest.warns(UserWarning, match="Expected"):
            strategy.ask(candidate_count=4)


@pytest.mark.parametrize("domain, experiments", [(domain, e) for e in [e3, e4]])
def test_strategy_ask_valid(
    domain: Domain,
    experiments: pd.DataFrame,
):
    strategy = dummy.DummyStrategy(
        data_model=dummy.DummyStrategyDataModel(domain=domain),
    )
    strategy.tell(experiments)

    def test_ask(self: Strategy, candidate_count: int):
        candidates = generate_candidates(self.domain, candidate_count)
        return candidates

    with mock.patch.object(dummy.DummyStrategy, "_ask", new=test_ask):
        strategy.ask(candidate_count=1)


def test_ask_invalid_candidate_count_request():
    strategy = dummy.DummyStrategy(
        data_model=dummy.DummyStrategyDataModel(domain=domain),
    )
    strategy.tell(e3)
    with pytest.raises(ValueError):
        strategy.ask(-1)


# test for PredictiveStrategy
@pytest.mark.parametrize(
    "domain, experiments",
    [
        (
            Domain.from_lists(
                inputs=[if1, if2],
                outputs=[of1, of2],
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
    strategy = dummy.DummyPredictiveStrategy(
        data_model=dummy.DummyPredictiveStrategyDataModel(domain=domain),
    )
    strategy.tell(experiments)

    def test_ask(self: Strategy, candidate_count: int):
        candidates = generate_candidates(self.domain, candidate_count)
        return candidates[domain.inputs.get_keys()]

    with mock.patch.object(dummy.DummyPredictiveStrategy, "_ask", new=test_ask):
        strategy.ask(candidate_count=1)


def test_predictivestrategy_to_candidates():
    domain = Domain.from_lists(
        inputs=[if1, if2],
        outputs=[of1, of2],
        constraints=[],
    )
    strategy = dummy.DummyPredictiveStrategy(
        data_model=dummy.DummyPredictiveStrategyDataModel(domain=domain),
    )
    candidates = generate_candidates(domain, 5)
    strategy.to_candidates(candidates=candidates)


@pytest.mark.parametrize(
    "domain, experiments",
    [
        (
            Domain.from_lists(
                inputs=[if1, if2],
                outputs=[of1, of2],
                constraints=[],
            ),
            e,
        )
        for e in [e3, e4]
    ],
)
def test_predictive_strategy_predict(domain, experiments):
    strategy = dummy.DummyPredictiveStrategy(
        data_model=dummy.DummyPredictiveStrategyDataModel(domain=domain),
    )
    strategy.tell(experiments)
    preds = strategy.predict(generate_candidates(domain=domain))
    assert sorted(preds.columns) == sorted(
        [
            "of1_pred",
            "of2_pred",
            "of1_sd",
            "of2_sd",
            "of1_des",
            "of2_des",
        ],
    )


@pytest.mark.parametrize(
    "domain",
    [
        (
            Domain.from_lists(
                inputs=[if1, if2],
                outputs=[of1, of2],
                constraints=[],
            )
        ),
    ],
)
def test_predictive_strategy_predict_not_ready(domain):
    strategy = dummy.DummyPredictiveStrategy(
        data_model=dummy.DummyPredictiveStrategyDataModel(domain=domain),
    )
    candidates = generate_candidates(domain=domain)
    with pytest.raises(ValueError, match="Model not yet fitted."):
        strategy.predict(candidates)
    strategy._is_fitted = True
    with pytest.raises(ValueError, match="No experiments available."):
        strategy.predict(candidates)
