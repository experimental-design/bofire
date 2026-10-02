import numpy as np
import pandas as pd
import pytest
import torch
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.acquisition.monte_carlo import qSimpleRegret, qUpperConfidenceBound
from botorch.acquisition.preference import qExpectedUtilityOfBestOption

from bofire.data_models.acquisition_functions.api import (
    qEUBO,
    qLogEI,
    qLogNEI,
    qSR,
    qUCB,
)
from bofire.data_models.constraints.api import LinearInequalityConstraint
from bofire.data_models.domain.api import Domain, Inputs, Outputs
from bofire.data_models.features.api import ContinuousInput, ContinuousOutput
from bofire.data_models.likelihoods.api import (
    PairwiseLogitLikelihood,
    PairwiseProbitLikelihood,
)
from bofire.data_models.objectives.api import MaximizeObjective, MinimizeObjective
from bofire.data_models.strategies.api import BotorchOptimizer
from bofire.data_models.strategies.api import BotorchStrategy as BotorchDataModel
from bofire.data_models.strategies.api import PreferenceStrategy as DataModel
from bofire.data_models.surrogates.api import (
    BotorchSurrogates,
    PairwiseGPSurrogate,
    SingleTaskGPSurrogate,
)
from bofire.strategies.api import PreferenceStrategy, map
from bofire.strategies.predictives.botorch import BotorchStrategy
from bofire.utils.torch_tools import tkwargs


def _domain(objective=None) -> Domain:
    return Domain(
        inputs=Inputs(features=[ContinuousInput(key="x", bounds=(0, 1))]),
        outputs=Outputs(
            features=[
                ContinuousOutput(
                    key="utility", objective=objective or MaximizeObjective()
                )
            ]
        ),
    )


def _data() -> tuple[pd.DataFrame, pd.DataFrame]:
    experiments = pd.DataFrame(
        {
            "x": [0.0, 0.25, 0.5, 0.75, 1.0],
            "labcode": ["a", "b", "c", "d", "e"],
        }
    )
    preferences = pd.DataFrame(
        {
            "labcode_A": ["b", "c", "d", "d"],
            "labcode_B": ["a", "b", "c", "e"],
            "preference": [1.0, 1.0, 1.0, 1.0],
        }
    )
    return experiments, preferences


def _strategy(acquisition_function=None, likelihood=None) -> PreferenceStrategy:
    if acquisition_function is None:
        acquisition_function = qEUBO(n_mc_samples=16)
    domain = _domain()
    return map(
        DataModel(
            domain=domain,
            surrogate_specs=BotorchSurrogates(
                surrogates=[
                    PairwiseGPSurrogate(
                        inputs=domain.inputs,
                        outputs=domain.outputs,
                        likelihood=likelihood or PairwiseProbitLikelihood(),
                    )
                ]
            ),
            acquisition_function=acquisition_function,
            acquisition_optimizer=BotorchOptimizer(
                n_restarts=2, n_raw_samples=32, maxiter=50
            ),
            seed=42,
        )
    )


def test_preference_strategy_data_model_defaults():
    data_model = DataModel(domain=_domain())

    assert isinstance(data_model, BotorchDataModel)
    assert len(data_model.surrogate_specs.surrogates) == 1
    surrogate = data_model.surrogate_specs.surrogates[0]
    assert isinstance(surrogate, PairwiseGPSurrogate)
    assert isinstance(data_model.acquisition_function, qEUBO)
    assert surrogate.inputs == data_model.domain.inputs
    assert surrogate.outputs == data_model.domain.outputs


@pytest.mark.parametrize(
    "likelihood", [PairwiseProbitLikelihood(), PairwiseLogitLikelihood()]
)
def test_preference_strategy_custom_surrogate_roundtrip(likelihood):
    domain = _domain()
    specs = BotorchSurrogates(
        surrogates=[
            PairwiseGPSurrogate(
                inputs=domain.inputs, outputs=domain.outputs, likelihood=likelihood
            )
        ]
    )
    strategy = PreferenceStrategy.make(domain=domain, surrogate_specs=specs)
    restored = DataModel.model_validate_json(strategy._data_model.model_dump_json())
    assert restored.surrogate_specs.model_dump(mode="json") == specs.model_dump(
        mode="json"
    )
    assert map(restored).surrogate.likelihood == likelihood


def test_preference_strategy_rejects_regression_surrogate():
    domain = _domain()
    specs = BotorchSurrogates(
        surrogates=[SingleTaskGPSurrogate(inputs=domain.inputs, outputs=domain.outputs)]
    )
    with pytest.raises(ValueError):
        DataModel(domain=domain, surrogate_specs=specs)


def test_preference_strategy_rejects_multiple_surrogates():
    domain = _domain()
    specs = BotorchSurrogates(
        surrogates=[
            PairwiseGPSurrogate(inputs=domain.inputs, outputs=domain.outputs),
            PairwiseGPSurrogate(
                inputs=domain.inputs,
                outputs=Outputs(features=[ContinuousOutput(key="other")]),
            ),
        ]
    )
    with pytest.raises(ValueError):
        DataModel(domain=domain, surrogate_specs=specs)


@pytest.mark.parametrize("mismatch", ["inputs", "outputs"])
def test_preference_strategy_rejects_mismatched_surrogate(mismatch):
    domain = _domain()
    surrogate_domain = _domain()
    if mismatch == "inputs":
        surrogate_domain.inputs.features[0].bounds = (0, 2)
    else:
        surrogate_domain.outputs.features[0].key = "other"
    specs = BotorchSurrogates(
        surrogates=[
            PairwiseGPSurrogate(
                inputs=surrogate_domain.inputs, outputs=surrogate_domain.outputs
            )
        ]
    )
    with pytest.raises(ValueError):
        DataModel(domain=domain, surrogate_specs=specs)


def test_preference_strategy_rejects_hyperparameter_tuning():
    with pytest.raises(ValueError, match="frequency_hyperopt"):
        DataModel(domain=_domain(), frequency_hyperopt=1)


@pytest.mark.parametrize("acquisition_class", [qEUBO, qLogNEI, qSR, qUCB])
def test_preference_strategy_data_model_accepts_acquisitions(acquisition_class):
    data_model = DataModel(
        domain=_domain(),
        acquisition_function=acquisition_class(n_mc_samples=16),
    )

    assert isinstance(data_model.acquisition_function, acquisition_class)
    reloaded = DataModel.model_validate_json(data_model.model_dump_json())
    assert reloaded.model_dump(mode="json") == data_model.model_dump(mode="json")


def test_preference_strategy_rejects_acquisition_requiring_observed_incumbent():
    with pytest.raises(ValueError):
        DataModel(domain=_domain(), acquisition_function=qLogEI())


def test_preference_strategy_requires_maximize_objective():
    with pytest.raises(ValueError, match="Objective .* is not implemented"):
        DataModel(domain=_domain(MinimizeObjective()))


def test_preference_strategy_requires_one_output():
    domain = _domain()
    domain.outputs.features.append(
        ContinuousOutput(key="other", objective=MaximizeObjective())
    )

    with pytest.raises(ValueError, match="exactly one PairwiseGPSurrogate"):
        DataModel(domain=domain)


@pytest.mark.parametrize("include_infeasible", [False, True])
def test_preference_baseline_uses_designs_without_observed_outputs(include_infeasible):
    domain = _domain()
    domain.inputs.features.append(ContinuousInput(key="y", bounds=(0, 1)))
    domain.constraints.constraints.append(
        LinearInequalityConstraint(
            features=["x", "y"], coefficients=[1.0, 1.0], rhs=0.95
        )
    )
    strategy = PreferenceStrategy.make(
        domain=domain,
        include_infeasible_exps_in_acqf_calc=include_infeasible,
    )
    experiments, preferences = _data()
    experiments = pd.concat(
        [experiments, pd.DataFrame({"x": [0.25], "labcode": ["duplicate_design"]})],
        ignore_index=True,
    )
    experiments = experiments.assign(y=0.2)[["x", "y", "labcode"]]
    strategy.tell(experiments, preferences=preferences, retrain=False)
    strategy.set_candidates(pd.DataFrame({"x": [0.33], "y": [0.2]}))

    X_train, X_pending = strategy.get_acqf_input_tensors()
    expected = (
        [0.0, 0.25, 0.5, 0.75, 1.0] if include_infeasible else [0.0, 0.25, 0.5, 0.75]
    )
    torch.testing.assert_close(X_train[:, 0], torch.tensor(expected, **tkwargs))
    torch.testing.assert_close(X_pending, torch.tensor([[0.33, 0.2]], **tkwargs))
    pd.testing.assert_frame_equal(strategy.experiments, experiments)


def test_preference_predictions_apply_fixed_objective_bounds():
    strategy = PreferenceStrategy.make(
        domain=_domain(MaximizeObjective(bounds=(-1.0, 3.0))), seed=42
    )
    experiments, preferences = _data()
    strategy.tell(experiments, preferences=preferences)
    predictions = strategy.predict(experiments)
    np.testing.assert_allclose(
        predictions["utility_des"], (predictions["utility_pred"] + 1.0) / 4.0
    )


def test_tell_requires_preferences():
    strategy = _strategy()
    experiments, _ = _data()

    with pytest.raises(ValueError, match="requires a `preferences` DataFrame"):
        strategy.tell(experiments)


def test_tell_appends_designs_and_comparisons():
    strategy = _strategy()
    experiments, preferences = _data()
    strategy.tell(experiments.iloc[:3], preferences=preferences.iloc[:2], retrain=False)
    strategy.tell(experiments.iloc[3:], preferences=preferences.iloc[2:], retrain=False)

    assert strategy.experiments is not None
    assert strategy.preferences is not None
    assert len(strategy.experiments) == 5
    assert len(strategy.preferences) == 4


def test_tell_appends_comparisons_without_new_designs():
    strategy = _strategy()
    experiments, preferences = _data()
    strategy.tell(experiments, preferences=preferences.iloc[:2], retrain=False)
    original_experiments = strategy.experiments.copy(deep=True)

    strategy.tell(pd.DataFrame(), preferences=preferences.iloc[2:], retrain=False)

    pd.testing.assert_frame_equal(strategy.experiments, original_experiments)
    pd.testing.assert_frame_equal(strategy.preferences, preferences)


def test_tell_replaces_designs_and_comparisons():
    strategy = _strategy()
    experiments, preferences = _data()
    strategy.tell(experiments, preferences=preferences, retrain=False)
    replacement_experiments = pd.DataFrame(
        {"x": [0.2, 0.6], "labcode": ["new_a", "new_b"]}
    )
    replacement_preferences = pd.DataFrame(
        {"labcode_A": ["new_a"], "labcode_B": ["new_b"], "preference": [-1.0]}
    )

    strategy.tell(
        replacement_experiments,
        preferences=replacement_preferences,
        replace=True,
        retrain=False,
    )

    pd.testing.assert_frame_equal(strategy.experiments, replacement_experiments)
    pd.testing.assert_frame_equal(strategy.preferences, replacement_preferences)


@pytest.mark.parametrize("replace", [False, True])
def test_invalid_feedback_leaves_both_tables_unchanged(replace):
    strategy = _strategy()
    experiments, preferences = _data()
    strategy.tell(experiments, preferences=preferences, retrain=False)
    original_experiments = strategy.experiments.copy(deep=True)
    original_preferences = strategy.preferences.copy(deep=True)

    with pytest.raises(ValueError, match="unknown labcodes"):
        strategy.tell(
            pd.DataFrame({"x": [0.1], "labcode": ["new"]}),
            preferences=pd.DataFrame(
                {
                    "labcode_A": ["new"],
                    "labcode_B": ["missing"],
                    "preference": [1.0],
                }
            ),
            replace=replace,
        )

    pd.testing.assert_frame_equal(strategy.experiments, original_experiments)
    pd.testing.assert_frame_equal(strategy.preferences, original_preferences)


def test_tell_rejects_duplicate_appended_labcode():
    strategy = _strategy()
    experiments, preferences = _data()
    strategy.tell(experiments.iloc[:3], preferences=preferences.iloc[:2], retrain=False)

    with pytest.raises(ValueError, match="Duplicate labcodes"):
        strategy.tell(
            experiments.iloc[[2]],
            preferences=pd.DataFrame(columns=preferences.columns),
            retrain=False,
        )


def test_tell_rejects_unknown_labcode():
    strategy = _strategy()
    experiments, preferences = _data()
    preferences.loc[0, "labcode_A"] = "unknown"

    with pytest.raises(ValueError, match="unknown labcodes"):
        strategy.tell(experiments, preferences=preferences, retrain=False)


@pytest.mark.parametrize(
    "invalid_preference",
    [float("nan"), float("inf"), -float("inf"), 0.5, 2.0],
)
def test_tell_rejects_invalid_preference(invalid_preference):
    strategy = _strategy()
    experiments, preferences = _data()
    preferences.loc[0, "preference"] = invalid_preference

    with pytest.raises(ValueError, match="Preference values must be one of"):
        strategy.tell(experiments, preferences=preferences, retrain=False)


@pytest.mark.parametrize(
    "acquisition, acquisition_type, likelihood, default_count",
    [
        (
            qEUBO(n_mc_samples=16),
            qExpectedUtilityOfBestOption,
            PairwiseProbitLikelihood(),
            2,
        ),
        (
            qLogNEI(n_mc_samples=16),
            qLogNoisyExpectedImprovement,
            PairwiseLogitLikelihood(),
            1,
        ),
        (qSR(n_mc_samples=16), qSimpleRegret, PairwiseProbitLikelihood(), 1),
        (
            qUCB(n_mc_samples=16, beta=0.4),
            qUpperConfidenceBound,
            PairwiseLogitLikelihood(),
            1,
        ),
    ],
    ids=["qEUBO", "qLogNEI", "qSR", "qUCB"],
)
def test_preference_strategy_fits_predicts_and_asks(
    acquisition, acquisition_type, likelihood, default_count
):
    strategy = _strategy(acquisition, likelihood)
    experiments, preferences = _data()
    strategy.tell(experiments, preferences=preferences)

    assert isinstance(strategy, BotorchStrategy)
    assert strategy.is_fitted
    assert "utility" not in strategy.experiments
    assert isinstance(strategy._get_acqfs(2)[0], acquisition_type)

    test_points = pd.DataFrame({"x": [0.3, 0.8]}, index=["left", "right"])
    predictions = strategy.predict(test_points)
    pd.testing.assert_index_equal(predictions.index, test_points.index)
    np.testing.assert_allclose(predictions["utility_des"], predictions["utility_pred"])
    assert np.isfinite(predictions.to_numpy()).all()

    if isinstance(acquisition, qEUBO):
        with pytest.raises(ValueError, match="at least two candidates"):
            strategy.ask(1)

    candidates = strategy.ask()
    assert len(candidates) == default_count
    assert {"x", "utility_pred", "utility_sd", "utility_des"}.issubset(
        candidates.columns
    )
    strategy.domain.validate_candidates(candidates)

    strategy.set_candidates(pd.DataFrame({"x": [0.15]}))
    _, X_pending = strategy.get_acqf_input_tensors()
    torch.testing.assert_close(X_pending, torch.tensor([[0.15]], **tkwargs))

    individual_values = strategy.calc_acquisition(test_points)
    batch_value = strategy.calc_acquisition(test_points, combined=True)
    assert individual_values.shape == (2,)
    assert batch_value.size == 1
    assert np.isfinite(individual_values).all()
    assert np.isfinite(batch_value).all()

    # Pending points must also work in the inherited batch optimizer path.
    batch_candidates = strategy.ask(2, add_pending=True)
    assert len(batch_candidates) == 2
    assert strategy.num_candidates == 3
    strategy.domain.validate_candidates(batch_candidates)


def test_preference_strategy_retains_ties_but_fits_only_non_tied_comparisons():
    strategy = _strategy()
    experiments, preferences = _data()
    ties = preferences.iloc[:1].assign(preference=0.0)
    strategy.tell(experiments, preferences=ties)

    assert not strategy.has_sufficient_experiments()
    assert not strategy.is_fitted
    pd.testing.assert_frame_equal(strategy.preferences, ties)
    with pytest.raises(ValueError, match="non-tied comparison"):
        strategy.fit()

    with pytest.warns(UserWarning, match=r"Dropping 1 pair\(s\)"):
        strategy.tell(pd.DataFrame(), preferences=preferences.iloc[1:2])

    assert strategy.has_sufficient_experiments()
    assert strategy.is_fitted
    assert len(strategy.preferences) == 2
    assert (strategy.preferences["preference"] == 0).sum() == 1
    torch.testing.assert_close(
        strategy.model.comparisons, torch.tensor([[2, 1]], dtype=torch.long)
    )

    # Replacing the observations with ties must invalidate the previously fitted
    # model even though there is no usable feedback to trigger a new fit.
    strategy.tell(experiments, preferences=ties, replace=True)
    assert not strategy.is_fitted
    assert not strategy.has_sufficient_experiments()
    with pytest.raises(ValueError, match="not yet fitted"):
        strategy.predict(experiments)
