import pandas as pd
import pytest
from botorch.acquisition.preference import qExpectedUtilityOfBestOption

from bofire.data_models.acquisition_functions.api import qEUBO
from bofire.data_models.domain.api import Domain, Inputs, Outputs
from bofire.data_models.features.api import ContinuousInput, ContinuousOutput
from bofire.data_models.objectives.api import MaximizeObjective, MinimizeObjective
from bofire.data_models.strategies.api import BotorchOptimizer
from bofire.data_models.strategies.api import PreferenceStrategy as DataModel
from bofire.data_models.surrogates.api import PairwiseGPSurrogate
from bofire.strategies.api import PreferenceStrategy, map


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


def _strategy() -> PreferenceStrategy:
    return map(
        DataModel(
            domain=_domain(),
            acquisition_function=qEUBO(n_mc_samples=16),
            acquisition_optimizer=BotorchOptimizer(
                n_restarts=2, n_raw_samples=32, maxiter=50
            ),
            seed=42,
        )
    )


def test_preference_strategy_data_model_defaults():
    data_model = DataModel(domain=_domain())

    assert isinstance(data_model.surrogate_spec, PairwiseGPSurrogate)
    assert isinstance(data_model.acquisition_function, qEUBO)
    assert data_model.surrogate_spec.inputs == data_model.domain.inputs
    assert data_model.surrogate_spec.outputs == data_model.domain.outputs


def test_preference_strategy_requires_maximize_objective():
    with pytest.raises(ValueError, match="Objective .* is not implemented"):
        DataModel(domain=_domain(MinimizeObjective()))


def test_preference_strategy_requires_one_output():
    domain = _domain()
    domain.outputs.features.append(
        ContinuousOutput(key="other", objective=MaximizeObjective())
    )

    with pytest.raises(ValueError, match="exactly one continuous latent utility"):
        DataModel(domain=domain)


def test_tell_requires_preferences():
    strategy = _strategy()
    experiments, _ = _data()

    with pytest.raises(ValueError, match="requires a `preferences` DataFrame"):
        strategy.tell(experiments)


def test_tell_preferences_appends_designs_and_comparisons():
    strategy = _strategy()
    experiments, preferences = _data()
    strategy.tell(experiments.iloc[:3], preferences=preferences.iloc[:2], retrain=False)
    strategy.tell(experiments.iloc[3:], preferences=preferences.iloc[2:], retrain=False)

    assert strategy.experiments is not None
    assert strategy.preferences is not None
    assert len(strategy.experiments) == 5
    assert len(strategy.preferences) == 4


def test_tell_preferences_rejects_unknown_labcode():
    strategy = _strategy()
    experiments, preferences = _data()
    preferences.loc[0, "labcode_A"] = "unknown"

    with pytest.raises(ValueError, match="unknown labcodes"):
        strategy.tell(experiments, preferences=preferences, retrain=False)


def test_tell_preferences_rejects_nonfinite_preference():
    strategy = _strategy()
    experiments, preferences = _data()
    preferences.loc[0, "preference"] = float("nan")

    with pytest.raises(ValueError, match="Preference values must be finite"):
        strategy.tell(experiments, preferences=preferences, retrain=False)


def test_preference_strategy_fits_qeubo_and_asks_for_pair():
    strategy = _strategy()
    experiments, preferences = _data()
    strategy.tell(experiments, preferences=preferences)

    assert strategy.is_fitted
    assert isinstance(strategy._get_acqf(), qExpectedUtilityOfBestOption)

    candidates = strategy.ask()
    assert len(candidates) == 2
    assert {"x", "utility_pred", "utility_sd", "utility_des"}.issubset(
        candidates.columns
    )
    strategy.domain.validate_candidates(candidates)


def test_preference_strategy_rejects_single_candidate():
    strategy = _strategy()
    experiments, preferences = _data()
    strategy.tell(experiments, preferences=preferences)

    with pytest.raises(ValueError, match="at least two candidates"):
        strategy.ask(1)
