import numpy as np
import pandas as pd
import pytest
import torch

from bofire.data_models.domain import api as domain_api
from bofire.data_models.molfeatures.api import Fingerprints
from bofire.data_models.surrogates.api import TanimotoGPSurrogate
from bofire.surrogates.api import map
from bofire.utils.torch_tools import tkwargs


@pytest.fixture(params=[1024, 2048])
def fingerprint_data_model(request) -> Fingerprints:
    return Fingerprints(
        bond_radius=3,
        n_bits=request.param
    )

def test_tanimoto_calculation(chem_domain_simple: tuple[domain_api.Domain, pd.DataFrame, pd.DataFrame],
                              fingerprint_data_model: Fingerprints):

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
    )

    surrogate1, surrogate2 = map(surrogate_data_model_no_pre_computation),\
        map(surrogate_data_model_with_pre_computation)
    surrogate1._fit(X, Y)
    surrogate2._fit(X, Y)

    # prediction: take molecules indeces 0, 2
    x = torch.from_numpy(np.array([0., 1.])).to(**tkwargs).reshape((-1, 1))

    #surrogate1.inputs.transform(X, specs=surrogate1.input_preprocessing_specs)
    fingerprints = surrogate1.model.input_transform(x)
    pred1 = surrogate1.model(fingerprints)
    pred2 = surrogate2.model(x)
    assert np.allclose(pred1.mean.detach().numpy(), pred2.mean.detach().numpy(), rtol=1e-3, atol=1e-3)
    assert np.allclose(pred1.variance.detach().numpy(), pred2.variance.detach().numpy(), rtol = 1e-3, atol = 1e-3)

    # compare mean
    mean1 = surrogate1.model.mean_module(fingerprints).detach().numpy()
    mean2 = surrogate2.model.mean_module(x).detach().numpy()
    assert np.allclose(mean1, mean2, rtol=1e-3, atol=1e-3)

    # compare covariance calculations
    cov1 = surrogate1.model.covar_module(fingerprints).detach().numpy()
    cov2 = surrogate2.model.covar_module(x).detach().numpy()
    assert np.allclose(cov1, cov2, rtol=1e-3, atol=1e-3)