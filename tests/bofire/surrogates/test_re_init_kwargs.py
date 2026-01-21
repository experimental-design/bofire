import pandas as pd

from bofire.data_models.domain import api as domain_api
from bofire.data_models.molfeatures.api import FingerprintsFragments
from bofire.data_models.surrogates.api import TanimotoGPSurrogate
from bofire.surrogates.api import map


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

    re_init_kwargs = surrogate.re_init_kwargs()  # includes the molecular fingerprints
    surrogate_re_init = map(surrogate_data_model, **re_init_kwargs)
    encodings_re_init = surrogate_re_init._input_transform.encoders[0].encoding
    assert (encodings == encodings_re_init).all()
