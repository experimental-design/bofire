import pytest
from pydantic import ValidationError

from bofire.data_models.domain.api import Inputs, Outputs
from bofire.data_models.features.api import ContinuousInput, ContinuousOutput
from bofire.data_models.likelihoods.api import (
    GaussianLikelihood,
    PairwiseLogitLikelihood,
    PairwiseProbitLikelihood,
)
from bofire.data_models.surrogates.api import PairwiseGPSurrogate, SingleTaskGPSurrogate


@pytest.fixture
def features():
    return {
        "inputs": Inputs(features=[ContinuousInput(key="x", bounds=[0, 1])]),
        "outputs": Outputs(features=[ContinuousOutput(key="utility")]),
    }


@pytest.mark.parametrize(
    "likelihood_cls",
    [PairwiseProbitLikelihood, PairwiseLogitLikelihood],
)
def test_pairwise_likelihood_serialization(features, likelihood_cls):
    surrogate = PairwiseGPSurrogate(**features, likelihood=likelihood_cls())
    assert surrogate.model_dump()["likelihood"] == likelihood_cls().model_dump()
    assert (
        PairwiseGPSurrogate.model_validate_json(surrogate.model_dump_json())
        == surrogate
    )


@pytest.mark.parametrize(
    "likelihood",
    [
        GaussianLikelihood(),
        GaussianLikelihood().model_dump(),
        "gaussian",
        "probit",
        "logit",
        {"type": "UnknownLikelihood"},
    ],
)
def test_pairwise_gp_rejects_incompatible_likelihood(features, likelihood):
    with pytest.raises(ValidationError):
        PairwiseGPSurrogate(**features, likelihood=likelihood)


@pytest.mark.parametrize(
    "likelihood",
    [
        PairwiseProbitLikelihood(),
        PairwiseLogitLikelihood(),
        PairwiseProbitLikelihood().model_dump(),
        PairwiseLogitLikelihood().model_dump(),
    ],
)
def test_single_task_gp_rejects_pairwise_likelihood(features, likelihood):
    with pytest.raises(ValidationError):
        SingleTaskGPSurrogate(**features, likelihood=likelihood)
