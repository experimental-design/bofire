import importlib

import pandas as pd
import pytest
import torch
from pandas.testing import assert_frame_equal

from bofire.data_models.descriptor_generators.api import Fingerprints, Fragments
from bofire.data_models.encodings.api import DescriptorEncoding
from bofire.data_models.kernels.api import ScaleKernel, TanimotoKernel
from bofire.data_models.likelihoods.api import THREESIX_LIKELIHOOD
from bofire.data_models.priors.api import THREESIX_SCALE_PRIOR
from bofire.data_models.surrogates.api import SingleTaskGPSurrogate
from bofire.surrogates.api import map


RDKIT_AVAILABLE = importlib.util.find_spec("rdkit") is not None


def _experiments(X: pd.DataFrame, Y: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([X, Y, pd.Series([1] * len(X), name="valid_output")], axis=1)


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
@pytest.mark.parametrize(
    "fingerprints",
    [
        DescriptorEncoding(columns=[], generators=[Fingerprints(n_bits=48)]),
        DescriptorEncoding(columns=[], generators=[Fragments()]),
        DescriptorEncoding(
            columns=[], generators=[Fingerprints(bond_radius=2, n_bits=24), Fragments()]
        ),
    ],
)
def test_precomputed_similarities_equal_on_the_fly(chem_domain_simple, fingerprints):
    domain, X, Y = chem_domain_simple
    precomputed = SingleTaskGPSurrogate.tanimoto(
        domain.inputs, domain.outputs, fingerprints=fingerprints
    )
    on_the_fly = SingleTaskGPSurrogate(
        inputs=domain.inputs,
        outputs=domain.outputs,
        categorical_encodings={"molecules": fingerprints},
        scaler=None,
        kernel=ScaleKernel(
            base_kernel=TanimotoKernel(features=["molecules"]),
            outputscale_prior=THREESIX_SCALE_PRIOR(),
        ),
        likelihood=THREESIX_LIKELIHOOD(),
    )

    predictions = []
    for data_model in [precomputed, on_the_fly]:
        torch.manual_seed(0)
        surrogate = map(data_model)
        surrogate.fit(_experiments(X, Y))
        predictions.append(surrogate.predict(X))

    assert_frame_equal(*predictions, rtol=1e-4)


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
def test_precomputed_similarities_need_ordinal_encoding(chem_domain_simple):
    domain, X, Y = chem_domain_simple
    fingerprints = DescriptorEncoding(columns=[], generators=[Fingerprints()])
    surrogate = map(
        SingleTaskGPSurrogate(
            inputs=domain.inputs,
            outputs=domain.outputs,
            categorical_encodings={"molecules": fingerprints},
            kernel=TanimotoKernel(features=["molecules"], fingerprints=fingerprints),
        )
    )

    with pytest.raises(ValueError, match="encoded as ordinal codes"):
        surrogate.fit(_experiments(X, Y))
