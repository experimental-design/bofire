import importlib

import pandas as pd
import pytest
import torch

from bofire.benchmarks.api import Himmelblau
from bofire.data_models.domain import api as domain_api
from bofire.data_models.molfeatures.api import FingerprintsFragments
from bofire.data_models.surrogates.api import SingleTaskGPSurrogate, TanimotoGPSurrogate
from bofire.surrogates.api import map


RDKIT_AVAILABLE = importlib.util.find_spec("rdkit") is not None


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
def test_re_init_kwargs_fingerprints(
    chem_domain_simple: tuple[domain_api.Domain, pd.DataFrame, pd.DataFrame],
):
    domain, X, Y = chem_domain_simple
    specs = {domain.inputs.get_keys()[0]: FingerprintsFragments(n_bits=2048)}
    surrogate_data_model = TanimotoGPSurrogate(
        inputs=domain.inputs,
        outputs=domain.outputs,
        categorical_encodings=specs,
    )

    surrogate = map(surrogate_data_model)
    assert surrogate._input_transform is None
    surrogate._fit(X=X, Y=Y)  # fitting triggers fingerprint computation
    encodings = surrogate._input_transform.encoders[0].encoding

    re_init_kwargs = surrogate.re_init_kwargs  # includes the molecular fingerprints
    surrogate_re_init = map(surrogate_data_model, **re_init_kwargs)
    encodings_re_init = surrogate_re_init._input_transform.encoders[0].encoding
    assert (encodings == encodings_re_init).all()


def test_SingleTaskGP_bound_relearning():
    bench = Himmelblau()
    experiments = bench.f(
        pd.DataFrame({"x_1": [0.0, 0.1], "x_2": [0.0, 0.1]}), return_complete=True
    )
    surrogate_data = SingleTaskGPSurrogate(
        inputs=bench.domain.inputs, outputs=bench.domain.outputs
    )
    surrogate = map(surrogate_data)
    surrogate.fit(experiments)
    bounds1 = surrogate.model.input_transform.bounds.clone()
    experiments2 = bench.f(bench.domain.inputs.sample(10), return_complete=True)
    surrogate.fit(experiments2)
    bounds2 = surrogate.model.input_transform.bounds.clone()
    assert not torch.equal(bounds1, bounds2)
