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

    # prediction: training data index
    x = torch.from_numpy(np.arange(3)).to(**tkwargs).reshape((-1, 1))

    #surrogate1.inputs.transform(X, specs=surrogate1.input_preprocessing_specs)
    surrogate1.model.forward(x)
    surrogate1.model.forward(surrogate1.model.input_transform(x))

    surrogate2.inputs.transform(X, specs=surrogate2.input_preprocessing_specs)
    surrogate2.model(x).mean