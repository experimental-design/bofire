import importlib

import gpytorch
import numpy as np
import pandas as pd
import pytest
import torch

from bofire.data_models.domain import api as domain_api
from bofire.data_models.molfeatures.api import (
    CompositeMolFeatures,
    Fingerprints,
    Fragments,
    MolFeatures,
)
from bofire.data_models.priors.api import GammaPrior
from bofire.data_models.strategies import api as strategies_api
from bofire.data_models.surrogates.api import BotorchSurrogates, TanimotoGPSurrogate
from bofire.strategies.api import map as map_strategy
from bofire.surrogates.api import map
from bofire.utils.torch_tools import tkwargs


RDKIT_AVAILABLE = importlib.util.find_spec("rdkit") is not None


@pytest.fixture(params=[24, 48])
def n_bits(request) -> int:
    return request.param


@pytest.fixture(params=["Figerprints", "Fragments", "Composite"])
def mol_feature_data_model(request, n_bits) -> MolFeatures:
    if request.param == "Figerprints":
        return Fingerprints(bond_radius=3, n_bits=n_bits)
    elif request.param == "Fragments":
        return Fragments()
    elif request.param == "Composite":
        return CompositeMolFeatures(
            features=[
                Fingerprints(bond_radius=2, n_bits=n_bits),
                Fragments(),
            ]
        )


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
def test_tanimoto_calculation(
    chem_domain_simple: tuple[domain_api.Domain, pd.DataFrame, pd.DataFrame],
    mol_feature_data_model: MolFeatures,
):
    domain, X, Y = chem_domain_simple

    surrogate_data_model_no_pre_computation = TanimotoGPSurrogate(
        inputs=domain.inputs,
        outputs=domain.outputs,
        tanimoto_calculation_mode="on_the_fly",
        categorical_encodings={domain.inputs.get_keys()[0]: mol_feature_data_model},
    )
    surrogate_data_model_with_pre_computation = TanimotoGPSurrogate(
        inputs=domain.inputs,
        outputs=domain.outputs,
        tanimoto_calculation_mode="pre_computed",
        categorical_encodings={domain.inputs.get_keys()[0]: mol_feature_data_model},
    )

    surrogate1, surrogate2 = (
        map(surrogate_data_model_no_pre_computation),
        map(surrogate_data_model_with_pre_computation),
    )
    surrogate1._fit(X, Y)
    surrogate2._fit(X, Y)

    # prediction: take molecules indeces 0, 2
    x = torch.from_numpy(
        surrogate2.inputs.transform(
            X, specs=surrogate2.input_preprocessing_specs
        ).values
    ).to(**tkwargs)

    # test predictions
    fingerprints = surrogate1.model.input_transform(x)
    prediction1 = surrogate1.model.forward(fingerprints)
    prediction2 = surrogate2.model.forward(x)
    mean1 = prediction1.mean.detach().numpy()
    cov1 = prediction1.covariance_matrix.detach().numpy()
    mean2 = prediction2.mean.detach().numpy()
    cov2 = prediction2.covariance_matrix.detach().numpy()

    assert np.allclose(mean1, mean2, rtol=1e-3, atol=1e-3)
    assert np.allclose(cov1, cov2)


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
def test_passing_of_tanimoto_sim_matrices(
    chem_domain_simple: tuple[domain_api.Domain, pd.DataFrame, pd.DataFrame],
):
    domain, X, Y = chem_domain_simple

    surrogate_data_model = TanimotoGPSurrogate(
        inputs=domain.inputs,
        outputs=domain.outputs,
        tanimoto_calculation_mode="pre_computed",
        categorical_encodings={domain.inputs.get_keys()[0]: Fingerprints()},
    )

    strategy_data_model = strategies_api.SoboStrategy(
        domain=domain,
        surrogate_specs=BotorchSurrogates(surrogates=[surrogate_data_model]),
    )

    strategy = map_strategy(strategy_data_model)

    strategy.tell(
        pd.concat((X, Y), axis=1)
    )  # computation of tanimoto distances happens here
    id_tensor_1 = id(
        strategy.surrogates.surrogates[
            0
        ].model.covar_module.base_kernel.tanimoto_similarity_matrix
    )

    strategy.tell(
        pd.concat((X, Y), axis=1)
    )  # computation of tanimoto distances happens here
    id_tensor_2 = id(
        strategy.surrogates.surrogates[
            0
        ].model.covar_module.base_kernel.tanimoto_similarity_matrix
    )

    assert (
        id_tensor_1 == id_tensor_2
    )  # passing matrix works would not change the id of the tensor


# --- Regression test for issue #762: noise_prior registration on TanimotoGPSurrogate ---


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
def test_noise_prior_registered_for_tanimoto_gp(chem_domain_simple):
    """The user-supplied ``noise_prior`` must end up in the fitted likelihood's
    ``_priors`` registry (which MLL reads from). Before the fix it was only set
    as an attribute and MLL silently used the BoTorch default.
    """
    torch.manual_seed(42)
    domain, X, Y = chem_domain_simple
    experiments = pd.concat(
        [X, Y, pd.Series([1] * len(X), name="valid_output")], axis=1
    )

    surrogate = map(
        TanimotoGPSurrogate(
            inputs=domain.inputs,
            outputs=domain.outputs,
            noise_prior=GammaPrior(concentration=1.1, rate=0.001),
            categorical_encodings={"molecules": Fingerprints()},
        )
    )
    surrogate.fit(experiments)

    priors = {n: p for n, _, p, _, _ in surrogate.model.likelihood.named_priors()}
    prior = priors.get("noise_covar.noise_prior")
    assert isinstance(prior, gpytorch.priors.GammaPrior), (
        f"User-supplied GammaPrior must be in the likelihood's _priors registry "
        f"(got {type(prior).__name__})"
    )
