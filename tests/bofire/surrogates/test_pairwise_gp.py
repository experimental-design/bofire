import numpy as np
import pandas as pd
import pytest
from botorch.models.likelihoods.pairwise import (
    PairwiseLogitLikelihood,
    PairwiseProbitLikelihood,
)
from botorch.models.pairwise_gp import PairwiseGP
from pandas.testing import assert_frame_equal
from scipy.stats import kendalltau

import bofire.surrogates.api as surrogates
from bofire.data_models.domain.api import Inputs, Outputs
from bofire.data_models.features.api import ContinuousInput, ContinuousOutput
from bofire.data_models.kernels.api import RBFKernel
from bofire.data_models.surrogates.api import PairwiseGPSurrogate


DIM = 3


def _utility(X: np.ndarray) -> np.ndarray:
    """Latent ground-truth utility: weighted sum of the inputs."""
    return X @ np.sqrt(np.arange(1, X.shape[1] + 1))


def _make_domain():
    inputs = Inputs(
        features=[
            ContinuousInput(key=f"x_{i + 1}", bounds=(0.0, 1.0)) for i in range(DIM)
        ]
    )
    outputs = Outputs(features=[ContinuousOutput(key="utility")])
    return inputs, outputs


def _make_data(n_points: int = 30, n_comparisons: int = 80, seed: int = 0):
    """Build (experiments, preferences) plus the true utility for scoring.

    Each comparison places the winner in slot A or B at random, so the *sign*
    of `preference` (not its slot) carries the label.
    """
    rng = np.random.default_rng(seed)
    X = rng.random((n_points, DIM))
    utility = _utility(X)
    labcodes = [f"cand_{i}" for i in range(n_points)]
    experiments = pd.DataFrame(X, columns=[f"x_{i + 1}" for i in range(DIM)])
    experiments["labcode"] = labcodes

    rows = []
    for _ in range(n_comparisons):
        i, j = rng.choice(n_points, size=2, replace=False)
        winner, loser = (i, j) if utility[i] > utility[j] else (j, i)
        if rng.random() < 0.5:
            rows.append((labcodes[winner], labcodes[loser], 1.0))  # A preferred
        else:
            rows.append((labcodes[loser], labcodes[winner], -1.0))  # B preferred
    preferences = pd.DataFrame(rows, columns=["labcode_A", "labcode_B", "preference"])
    return experiments, preferences, utility


def test_pairwise_gp_fit_and_predict():
    inputs, outputs = _make_domain()
    experiments, preferences, utility = _make_data()

    surrogate = surrogates.map(PairwiseGPSurrogate(inputs=inputs, outputs=outputs))

    # predicting before fitting raises
    with pytest.raises(ValueError, match="not fitted"):
        surrogate.predict(experiments[[f"x_{i + 1}" for i in range(DIM)]])

    surrogate.fit(experiments, preferences)
    assert surrogate.is_fitted
    assert isinstance(surrogate.model, PairwiseGP)

    preds = surrogate.predict(experiments[[f"x_{i + 1}" for i in range(DIM)]])
    assert list(preds.columns) == ["utility_pred", "utility_sd"]
    assert len(preds) == len(experiments)
    assert preds["utility_sd"].ge(0).all()
    assert np.isfinite(preds.to_numpy()).all()

    # the surrogate should recover the latent utility ranking
    corr = kendalltau(preds["utility_pred"].to_numpy(), utility).correlation
    assert corr > 0.8


def test_pairwise_gp_sign_convention_is_consistent():
    """Flipping every A/B slot and every preference sign must not change the fit."""
    inputs, outputs = _make_domain()
    experiments, preferences, _ = _make_data()

    flipped = pd.DataFrame(
        {
            "labcode_A": preferences["labcode_B"],
            "labcode_B": preferences["labcode_A"],
            "preference": -preferences["preference"],
        }
    )

    test_X = experiments[[f"x_{i + 1}" for i in range(DIM)]]

    surrogate = surrogates.map(PairwiseGPSurrogate(inputs=inputs, outputs=outputs))
    surrogate.fit(experiments, preferences)
    preds = surrogate.predict(test_X)

    surrogate_flipped = surrogates.map(
        PairwiseGPSurrogate(inputs=inputs, outputs=outputs)
    )
    surrogate_flipped.fit(experiments, flipped)
    preds_flipped = surrogate_flipped.predict(test_X)

    assert_frame_equal(preds, preds_flipped)


def test_pairwise_gp_dump_and_load():
    inputs, outputs = _make_domain()
    experiments, preferences, _ = _make_data()
    test_X = experiments[[f"x_{i + 1}" for i in range(DIM)]]

    surrogate = surrogates.map(PairwiseGPSurrogate(inputs=inputs, outputs=outputs))
    surrogate.fit(experiments, preferences)
    preds = surrogate.predict(test_X)
    dump = surrogate.dumps()

    reloaded = surrogates.map(PairwiseGPSurrogate(inputs=inputs, outputs=outputs))
    reloaded.loads(dump)
    assert reloaded.is_fitted
    assert_frame_equal(preds, reloaded.predict(test_X))


def test_pairwise_gp_requires_labcode_column():
    inputs, outputs = _make_domain()
    experiments, preferences, _ = _make_data()
    surrogate = surrogates.map(PairwiseGPSurrogate(inputs=inputs, outputs=outputs))
    with pytest.raises(ValueError, match="labcode"):
        surrogate.fit(experiments.drop(columns=["labcode"]), preferences)


def test_pairwise_gp_rejects_duplicate_labcodes():
    inputs, outputs = _make_domain()
    experiments, preferences, _ = _make_data()
    experiments.loc[1, "labcode"] = experiments.loc[0, "labcode"]
    surrogate = surrogates.map(PairwiseGPSurrogate(inputs=inputs, outputs=outputs))
    with pytest.raises(ValueError, match="Duplicate labcodes"):
        surrogate.fit(experiments, preferences)


def test_pairwise_gp_rejects_missing_preference_columns():
    inputs, outputs = _make_domain()
    experiments, preferences, _ = _make_data()
    surrogate = surrogates.map(PairwiseGPSurrogate(inputs=inputs, outputs=outputs))
    with pytest.raises(ValueError, match="missing required columns"):
        surrogate.fit(experiments, preferences.drop(columns=["preference"]))


def test_pairwise_gp_rejects_unknown_labcode():
    inputs, outputs = _make_domain()
    experiments, preferences, _ = _make_data()
    preferences.loc[0, "labcode_A"] = "does_not_exist"
    surrogate = surrogates.map(PairwiseGPSurrogate(inputs=inputs, outputs=outputs))
    with pytest.raises(ValueError, match="not present in experiments"):
        surrogate.fit(experiments, preferences)


def test_pairwise_gp_drops_ties_with_warning():
    inputs, outputs = _make_domain()
    experiments, preferences, _ = _make_data()
    preferences.loc[0, "preference"] = 0.0
    surrogate = surrogates.map(PairwiseGPSurrogate(inputs=inputs, outputs=outputs))
    with pytest.warns(UserWarning, match="ties"):
        surrogate.fit(experiments, preferences)
    assert surrogate.is_fitted


def test_pairwise_gp_all_ties_raises():
    inputs, outputs = _make_domain()
    experiments, preferences, _ = _make_data()
    preferences["preference"] = 0.0
    surrogate = surrogates.map(PairwiseGPSurrogate(inputs=inputs, outputs=outputs))
    with pytest.warns(UserWarning, match="ties"):
        with pytest.raises(ValueError, match="No valid pairs"):
            surrogate.fit(experiments, preferences)


def test_pairwise_gp_data_model_validation():
    inputs, _ = _make_domain()

    # exactly one output is required
    with pytest.raises(ValueError, match="exactly one output"):
        PairwiseGPSurrogate(
            inputs=inputs,
            outputs=Outputs(
                features=[ContinuousOutput(key="y1"), ContinuousOutput(key="y2")]
            ),
        )

    # the kernel must be a ScaleKernel
    with pytest.raises(ValueError, match="must be a ScaleKernel"):
        PairwiseGPSurrogate(
            inputs=inputs,
            outputs=Outputs(features=[ContinuousOutput(key="utility")]),
            kernel=RBFKernel(ard=True),
        )


@pytest.mark.parametrize(
    "likelihood, expected_cls",
    [
        ("probit", PairwiseProbitLikelihood),
        ("logit", PairwiseLogitLikelihood),
    ],
)
def test_pairwise_gp_likelihood(likelihood, expected_cls):
    """Both pairwise likelihoods fit and map to the right BoTorch class."""
    inputs, outputs = _make_domain()
    experiments, preferences, utility = _make_data()

    surrogate = surrogates.map(
        PairwiseGPSurrogate(inputs=inputs, outputs=outputs, likelihood=likelihood)
    )
    surrogate.fit(experiments, preferences)

    assert isinstance(surrogate.model.likelihood, expected_cls)

    preds = surrogate.predict(experiments[[f"x_{i + 1}" for i in range(DIM)]])
    corr = kendalltau(preds["utility_pred"].to_numpy(), utility).correlation
    assert corr > 0.8


def test_pairwise_gp_likelihood_default_is_probit():
    inputs, outputs = _make_domain()
    assert PairwiseGPSurrogate(inputs=inputs, outputs=outputs).likelihood == "probit"
