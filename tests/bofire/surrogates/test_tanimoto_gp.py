import importlib

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
    #mol_feature_data_model: MolFeatures,
):
    domain, X, Y = chem_domain_simple

    mol_feature_data_model = Fingerprints(bond_radius=3, n_bits=24)

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
    mean1 = surrogate1.model.mean_module(fingerprints).detach().numpy()
    cov1 = surrogate1.model.covar_module(fingerprints).detach().numpy()


    mean2 = surrogate2.model.mean_module(x).detach().numpy()
    cov2 = surrogate2.model.covar_module(x).detach().numpy()


    assert np.allclose(mean1, mean2)
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
        strategy.surrogates.surrogates[0].model.covar_module.base_kernel.sim_matrices[
            "molecules"
        ]
    )

    strategy.tell(
        pd.concat((X, Y), axis=1)
    )  # computation of tanimoto distances happens here
    id_tensor_2 = id(
        strategy.surrogates.surrogates[0].model.covar_module.base_kernel.sim_matrices[
            "molecules"
        ]
    )

    assert (
        id_tensor_1 == id_tensor_2
    )  # passing matrix works would not change the id of the tensor