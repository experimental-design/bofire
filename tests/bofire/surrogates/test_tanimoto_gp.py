import importlib

import numpy as np
import pandas as pd
import pytest
import torch

from bofire.data_models.domain import api as domain_api
from bofire.data_models.molfeatures.api import Fingerprints
from bofire.data_models.surrogates.api import TanimotoGPSurrogate
from bofire.surrogates.api import map
from bofire.utils.torch_tools import tkwargs


RDKIT_AVAILABLE = importlib.util.find_spec("rdkit") is not None


@pytest.fixture(params=[1024, 2048])
def fingerprint_data_model(request) -> Fingerprints:
    return Fingerprints(bond_radius=3, n_bits=request.param)

@pytest.mark.parametrize('chem_domain_simple', [False, True], indirect=True)  # multi-component
@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
def test_tanimoto_calculation(
    chem_domain_simple: tuple[domain_api.Domain, pd.DataFrame, pd.DataFrame],
    fingerprint_data_model: Fingerprints,
):
    domain, X, Y = chem_domain_simple

    surrogate_data_model_no_pre_computation = TanimotoGPSurrogate(
        inputs=domain.inputs,
        outputs=domain.outputs,
        pre_compute_similarities=False,
        categorical_encodings={domain.inputs.get_keys()[0]: fingerprint_data_model},
    )
    surrogate_data_model_with_pre_computation = TanimotoGPSurrogate(
        inputs=domain.inputs,
        outputs=domain.outputs,
        pre_compute_similarities=True,
        categorical_encodings={domain.inputs.get_keys()[0]: fingerprint_data_model},
    )

    surrogate1, surrogate2 = (
        map(surrogate_data_model_no_pre_computation),
        map(surrogate_data_model_with_pre_computation),
    )
    surrogate2._fit(X, Y)

    # prediction: take molecules indeces 0, 2
    x = torch.from_numpy(
        surrogate2.inputs.transform(X, specs=surrogate2.input_preprocessing_specs).values).to(**tkwargs)

    # test predictions
    pred2 = surrogate2.model(x)
    mean2 = surrogate2.model.mean_module(x).detach().numpy()
    cov2 = surrogate2.model.covar_module(x).detach().numpy()

    if len(X.columns) == 1:  # for the single-input case, we can compare with the ad-hoc calculated similarities

        surrogate1._fit(X, Y)
        fingerprints = surrogate1.model.input_transform(x)
        pred1 = surrogate1.model(fingerprints)
        mean1 = surrogate1.model.mean_module(fingerprints).detach().numpy()
        cov1 = surrogate1.model.covar_module(fingerprints).detach().numpy()

        assert np.allclose(
            pred1.mean.detach().numpy(), pred2.mean.detach().numpy(), rtol=1e-3, atol=1e-3
        )
        assert np.allclose(
            pred1.variance.detach().numpy(),
            pred2.variance.detach().numpy(),
            rtol=1e-3,
            atol=1e-3,
        )
        assert np.allclose(mean1, mean2, rtol=1e-3, atol=1e-3)
        assert np.allclose(cov1, cov2, rtol=1e-3, atol=1e-3)
