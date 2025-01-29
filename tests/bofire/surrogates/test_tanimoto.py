import importlib

import pandas as pd
import pytest
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import (
    ChainedInputTransform,
    InputStandardize,
    Normalize,
    OneHotToNumeric,
)
from botorch.models.transforms.outcome import Standardize
from pandas.testing import assert_frame_equal
from pydantic import ValidationError

import bofire.surrogates.api as surrogates
from bofire.data_models.domain.api import Inputs, Outputs
from bofire.data_models.enum import CategoricalEncodingEnum
from bofire.data_models.features.api import (
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
    MolecularInput,
)
from bofire.data_models.kernels.api import (
    MaternKernel,
    RBFKernel,
    ScaleKernel,
    TanimotoKernel,
)
from bofire.data_models.molfeatures.api import (
    Fingerprints,
    FingerprintsFragments,
    Fragments,
    MordredDescriptors,
)
from bofire.data_models.surrogates.api import ScalerEnum
from bofire.data_models.surrogates.mixed_tanimoto_gp import MixedTanimotoGPSurrogate
from bofire.data_models.surrogates.tanimoto_gp import TanimotoGPSurrogate
from bofire.surrogates.mixed_tanimoto_gp import MixedTanimotoGP


RDKIT_AVAILABLE = importlib.util.find_spec("rdkit") is not None


def test_TanimotoGPModel_invalid_preprocessing_mordred():
    inputs = Inputs(features=[MolecularInput(key="x_mol")])
    outputs = Outputs(features=[ContinuousOutput(key="y")])
    experiments = [
        ["CC(=O)Oc1ccccc1C(=O)O", 1.5],
        ["c1ccccc1", 3.5],
        ["[CH3][CH2][OH]", 2.0],
        ["N[C@](C)(F)C(=O)O", 4.0],
    ]
    experiments = pd.DataFrame(experiments, columns=["x_mol", "y"])
    experiments["valid_y"] = 1
    with pytest.raises(
        ValidationError,
        match="TanimotoGPSurrogate can only be used if at least one of fingerprints, fragments, or fingerprintsfragments features are present.",
    ):
        TanimotoGPSurrogate(
            inputs=inputs,
            outputs=outputs,
            input_preprocessing_specs={
                "x_mol": MordredDescriptors(descriptors=["NssCH2", "ATSC2d"]),
            },
        )


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
@pytest.mark.parametrize(
    "kernel, specs",
    [
        (
            ScaleKernel(base_kernel=TanimotoKernel(ard=False)),
            {"x_1": Fingerprints(n_bits=32)},
        ),
        (
            ScaleKernel(base_kernel=TanimotoKernel(ard=True)),
            {"x_1": Fragments()},
        ),
        (
            ScaleKernel(base_kernel=TanimotoKernel(ard=False)),
            {"x_1": FingerprintsFragments(n_bits=32)},
        ),
    ],
)
def test_TanimotoGP(kernel, specs):
    inputs = Inputs(features=[MolecularInput(key="x_1")])
    outputs = Outputs(features=[ContinuousOutput(key="y")])
    experiments = [
        ["CC(=O)Oc1ccccc1C(=O)O", 88.0],
        ["c1ccccc1", 35.0],
        ["[CH3][CH2][OH]", 69.0],
        ["N[C@](C)(F)C(=O)O", 20.0],
    ]
    experiments = pd.DataFrame(experiments, columns=["x_1", "y"])
    experiments["valid_y"] = 1
    model = TanimotoGPSurrogate(
        inputs=inputs,
        outputs=outputs,
        kernel=kernel,
        input_preprocessing_specs=specs,
    )
    model = surrogates.map(model)
    model.fit(experiments)
    # dump the model
    dump = model.dumps()
    # make predictions
    preds = model.predict(experiments.iloc[:-1])
    assert preds.shape == (3, 2)
    # check that model is composed correctly
    assert isinstance(model.model, SingleTaskGP)
    assert isinstance(model.model.outcome_transform, Standardize)
    assert model.is_compatibilized is False
    # reload the model from dump and check for equality in predictions
    model2 = TanimotoGPSurrogate(
        inputs=inputs,
        outputs=outputs,
        kernel=kernel,
        input_preprocessing_specs=specs,
    )
    model2 = surrogates.map(model2)
    model2.loads(dump)
    preds2 = model2.predict(experiments.iloc[:-1])
    assert_frame_equal(preds, preds2)


def test_MixedTanimotoGPModel_invalid_preprocessing():
    inputs = Inputs(
        features=[
            ContinuousInput(
                key=f"x_{i+1}",
                bounds=(-4, 4),
            )
            for i in range(2)
        ]
        + [CategoricalInput(key="x_cat", categories=["mama", "papa"])],
    )
    outputs = Outputs(features=[ContinuousOutput(key="y")])
    experiments = inputs.sample(n=10)
    experiments.eval("y=((x_1**2 + x_2 - 11)**2+(x_1 + x_2**2 -7)**2)", inplace=True)
    experiments.loc[experiments.x_cat == "mama", "y"] *= 5.0
    experiments.loc[experiments.x_cat == "papa", "y"] /= 2.0
    experiments["valid_y"] = 1
    with pytest.raises(
        ValidationError,
        match="MixedTanimotoGPSurrogate can only be used if at least one of fingerprints, fragments, or fingerprintsfragments features are present.",
    ):
        MixedTanimotoGPSurrogate(
            inputs=inputs,
            outputs=outputs,
        )


def test_MixedTanimotoGPModel_invalid_preprocessing_mordred():
    inputs = Inputs(
        features=[
            ContinuousInput(
                key=f"x_{i+1}",
                bounds=(0, 5),
            )
            for i in range(2)
        ]
        + [MolecularInput(key="x_mol")],
    )
    outputs = Outputs(features=[ContinuousOutput(key="y")])
    experiments = [
        [5.0, 2.5, "CC(=O)Oc1ccccc1C(=O)O", 1.5],
        [4.0, 2.0, "c1ccccc1", 3.5],
        [3.0, 0.5, "[CH3][CH2][OH]", 2.0],
        [1.5, 4.5, "N[C@](C)(F)C(=O)O", 4.0],
    ]
    experiments = pd.DataFrame(experiments, columns=["x_1", "x_2", "x_mol", "y"])
    experiments["valid_y"] = 1
    with pytest.raises(
        ValidationError,
        match="MixedTanimotoGPSurrogate can only be used if at least one of fingerprints, fragments, or fingerprintsfragments features are present.",
    ):
        MixedTanimotoGPSurrogate(
            inputs=inputs,
            outputs=outputs,
            input_preprocessing_specs={
                "x_mol": MordredDescriptors(descriptors=["NssCH2", "ATSC2d"]),
            },
        )


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
@pytest.mark.parametrize(
    "kernel, specs, scaler",
    [
        (
            TanimotoKernel(ard=True),
            {"x_mol": Fingerprints(n_bits=32)},
            ScalerEnum.NORMALIZE,
        ),
        (
            TanimotoKernel(ard=False),
            {"x_mol": Fragments()},
            ScalerEnum.IDENTITY,
        ),
        (
            TanimotoKernel(ard=True),
            {"x_mol": FingerprintsFragments(n_bits=32)},
            ScalerEnum.STANDARDIZE,
        ),
    ],
)
def test_MixedTanimotoGP_continuous(kernel, specs, scaler):
    inputs = Inputs(
        features=[MolecularInput(key="x_mol")]
        + [
            ContinuousInput(
                key=f"x_{i+1}",
                bounds=(0, 5.0),
            )
            for i in range(2)
        ],
    )
    outputs = Outputs(features=[ContinuousOutput(key="y")])
    experiments = [
        ["CC(=O)Oc1ccccc1C(=O)O", 5.0, 2.5, 88.0],
        ["c1ccccc1", 4.0, 2.0, 35.0],
        ["[CH3][CH2][OH]", 3.0, 0.5, 69.0],
        ["N[C@](C)(F)C(=O)O", 1.5, 4.5, 20.0],
    ]
    experiments = pd.DataFrame(experiments, columns=["x_mol", "x_1", "x_2", "y"])
    experiments["valid_y"] = 1
    model = MixedTanimotoGPSurrogate(
        inputs=inputs,
        outputs=outputs,
        molecular_kernel=kernel,
        scaler=scaler,
        input_preprocessing_specs=specs,
    )
    model = surrogates.map(model)
    model.fit(experiments)
    # dump the model
    dump = model.dumps()
    # make predictions
    preds = model.predict(experiments.iloc[:-1])
    assert preds.shape == (3, 2)
    # check that model is composed correctly
    assert isinstance(model.model, MixedTanimotoGP)
    assert isinstance(model.model.outcome_transform, Standardize)
    if scaler == ScalerEnum.NORMALIZE:
        assert isinstance(model.model.input_transform, Normalize)
        assert model.model.input_transform.indices.shape == torch.Size([2])
    elif scaler == ScalerEnum.STANDARDIZE:
        assert isinstance(model.model.input_transform, InputStandardize)
        assert model.model.input_transform.indices.shape == torch.Size([2])
    else:
        with pytest.raises(
            AttributeError,
            match="'MixedTanimotoGP' object has no attribute 'input_transform'",
        ):
            assert model.model.input_transform is None
    assert model.is_compatibilized is False
    # reload the model from dump and check for equality in predictions
    model2 = MixedTanimotoGPSurrogate(
        inputs=inputs,
        outputs=outputs,
        molecular_kernel=kernel,
        input_preprocessing_specs=specs,
    )
    model2 = surrogates.map(model2)
    model2.loads(dump)
    preds2 = model2.predict(experiments.iloc[:-1])
    assert_frame_equal(preds, preds2)


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
@pytest.mark.parametrize(
    "kernel, specs, scaler",
    [
        (
            TanimotoKernel(ard=True),
            {
                "x_mol": Fingerprints(n_bits=32),
                "x_cat": CategoricalEncodingEnum.ONE_HOT,
            },
            ScalerEnum.NORMALIZE,
        ),
        (
            TanimotoKernel(ard=False),
            {"x_mol": Fragments(), "x_cat": CategoricalEncodingEnum.ONE_HOT},
            ScalerEnum.IDENTITY,
        ),
        (
            TanimotoKernel(ard=True),
            {
                "x_mol": FingerprintsFragments(n_bits=32),
                "x_cat": CategoricalEncodingEnum.ONE_HOT,
            },
            ScalerEnum.STANDARDIZE,
        ),
    ],
)
def test_MixedTanimotoGP(kernel, specs, scaler):
    inputs = Inputs(
        features=[MolecularInput(key="x_mol")]
        + [
            ContinuousInput(
                key="x_1",
                bounds=(0, 5.0),
            ),
        ]
        + [CategoricalInput(key="x_cat", categories=["a", "b"])],
    )
    outputs = Outputs(features=[ContinuousOutput(key="y")])
    experiments = [
        ["CC(=O)Oc1ccccc1C(=O)O", 5.0, "a", 88.0],
        ["c1ccccc1", 4.0, "a", 35.0],
        ["[CH3][CH2][OH]", 3.0, "b", 69.0],
        ["N[C@](C)(F)C(=O)O", 1.5, "b", 20.0],
    ]
    experiments = pd.DataFrame(experiments, columns=["x_mol", "x_1", "x_cat", "y"])
    experiments["valid_y"] = 1
    model = MixedTanimotoGPSurrogate(
        inputs=inputs,
        outputs=outputs,
        molecular_kernel=kernel,
        scaler=scaler,
        input_preprocessing_specs=specs,
    )
    model = surrogates.map(model)
    model.fit(experiments)
    # dump the model
    dump = model.dumps()
    # make predictions
    preds = model.predict(experiments.iloc[:-1])
    assert preds.shape == (3, 2)
    # check that model is composed correctly
    assert isinstance(model.model, MixedTanimotoGP)
    assert isinstance(model.model.outcome_transform, Standardize)
    if scaler == ScalerEnum.NORMALIZE:
        assert isinstance(model.model.input_transform, ChainedInputTransform)
        assert isinstance(model.model.input_transform.tf1, Normalize)
        assert model.model.input_transform.tf1.indices.shape == torch.Size([1])
        assert isinstance(model.model.input_transform.tf2, OneHotToNumeric)
    elif scaler == ScalerEnum.STANDARDIZE:
        assert isinstance(model.model.input_transform, ChainedInputTransform)
        assert isinstance(model.model.input_transform.tf1, InputStandardize)
        assert model.model.input_transform.tf1.indices.shape == torch.Size([1])
        assert isinstance(model.model.input_transform.tf2, OneHotToNumeric)
    else:
        assert isinstance(model.model.input_transform, OneHotToNumeric)
    assert model.is_compatibilized is False
    # reload the model from dump and check for equality in predictions
    model2 = MixedTanimotoGPSurrogate(
        inputs=inputs,
        outputs=outputs,
        molecular_kernel=kernel,
        input_preprocessing_specs=specs,
    )
    model2 = surrogates.map(model2)
    model2.loads(dump)
    preds2 = model2.predict(experiments.iloc[:-1])
    assert_frame_equal(preds, preds2)


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
@pytest.mark.parametrize(
    "kernel, specs",
    [
        (
            TanimotoKernel(ard=True),
            {
                "x_mol": Fingerprints(n_bits=32),
                "x_cat": CategoricalEncodingEnum.ONE_HOT,
            },
        ),
        (
            TanimotoKernel(ard=False),
            {"x_mol": Fragments(), "x_cat": CategoricalEncodingEnum.ONE_HOT},
        ),
        (
            TanimotoKernel(ard=True),
            {
                "x_mol": FingerprintsFragments(n_bits=32),
                "x_cat": CategoricalEncodingEnum.ONE_HOT,
            },
        ),
    ],
)
def test_MixedTanimotoGP_categorical(kernel, specs):
    inputs = Inputs(
        features=[MolecularInput(key="x_mol")]
        + [CategoricalInput(key="x_cat", categories=["a", "b"])],
    )
    outputs = Outputs(features=[ContinuousOutput(key="y")])
    experiments = [
        ["CC(=O)Oc1ccccc1C(=O)O", "a", 88.0],
        ["c1ccccc1", "a", 35.0],
        ["[CH3][CH2][OH]", "b", 69.0],
        ["N[C@](C)(F)C(=O)O", "b", 20.0],
    ]
    experiments = pd.DataFrame(experiments, columns=["x_mol", "x_cat", "y"])
    experiments["valid_y"] = 1
    model = MixedTanimotoGPSurrogate(
        inputs=inputs,
        outputs=outputs,
        molecular_kernel=kernel,
        input_preprocessing_specs=specs,
    )
    model = surrogates.map(model)
    model.fit(experiments)
    # dump the model
    dump = model.dumps()
    # make predictions
    preds = model.predict(experiments.iloc[:-1])
    assert preds.shape == (3, 2)
    # check that model is composed correctly
    assert isinstance(model.model, MixedTanimotoGP)
    assert isinstance(model.model.outcome_transform, Standardize)
    assert isinstance(model.model.input_transform, OneHotToNumeric)
    assert model.is_compatibilized is False
    # reload the model from dump and check for equality in predictions
    model2 = MixedTanimotoGPSurrogate(
        inputs=inputs,
        outputs=outputs,
        input_preprocessing_specs=specs,
    )
    model2 = surrogates.map(model2)
    model2.loads(dump)
    preds2 = model2.predict(experiments.iloc[:-1])
    assert_frame_equal(preds, preds2)


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
@pytest.mark.parametrize(
    "molecular_kernel, continuous_kernel, specs, scaler",
    [
        (
            TanimotoKernel(ard=True),
            RBFKernel(ard=True),
            {
                "x_1": Fingerprints(n_bits=32),
                "x_2": MordredDescriptors(descriptors=["NssCH2", "ATSC2d"]),
            },
            ScalerEnum.NORMALIZE,
        ),
        (
            TanimotoKernel(ard=False),
            MaternKernel(nu=0.5, ard=True),
            {
                "x_1": Fragments(),
                "x_2": MordredDescriptors(descriptors=["NssCH2", "ATSC2d"]),
            },
            ScalerEnum.IDENTITY,
        ),
        (
            TanimotoKernel(ard=True),
            MaternKernel(nu=2.5, ard=False),
            {
                "x_1": FingerprintsFragments(n_bits=32),
                "x_2": MordredDescriptors(descriptors=["NssCH2", "ATSC2d"]),
            },
            ScalerEnum.STANDARDIZE,
        ),
    ],
)
def test_MixedTanimotoGP_with_mordred(
    molecular_kernel,
    continuous_kernel,
    specs,
    scaler,
):
    inputs = Inputs(
        features=[
            MolecularInput(
                key=f"x_{i+1}",
            )
            for i in range(2)
        ],
    )
    outputs = Outputs(features=[ContinuousOutput(key="y")])
    experiments = [
        ["CC(=O)Oc1ccccc1C(=O)O", "CC(=O)Oc1ccccc1C(=O)O", 88.0],
        ["c1ccccc1", "c1ccccc1", 35.0],
        ["[CH3][CH2][OH]", "[CH3][CH2][OH]", 69.0],
        ["N[C@](C)(F)C(=O)O", "N[C@](C)(F)C(=O)O", 20.0],
    ]
    experiments = pd.DataFrame(experiments, columns=["x_1", "x_2", "y"])
    experiments["valid_y"] = 1
    model = MixedTanimotoGPSurrogate(
        inputs=inputs,
        outputs=outputs,
        molecular_kernel=molecular_kernel,
        continuous_kernel=continuous_kernel,
        scaler=scaler,
        input_preprocessing_specs=specs,
    )
    model = surrogates.map(model)
    model.fit(experiments)
    # dump the model
    dump = model.dumps()
    # make predictions
    preds = model.predict(experiments.iloc[:-1])
    assert preds.shape == (3, 2)
    # check that model is composed correctly
    assert isinstance(model.model, MixedTanimotoGP)
    assert isinstance(model.model.outcome_transform, Standardize)
    if scaler == ScalerEnum.NORMALIZE:
        assert isinstance(model.model.input_transform, Normalize)
        assert model.model.input_transform.indices.shape == torch.Size([2])
    elif scaler == ScalerEnum.STANDARDIZE:
        assert isinstance(model.model.input_transform, InputStandardize)
        assert model.model.input_transform.indices.shape == torch.Size([2])
    else:
        with pytest.raises(
            AttributeError,
            match="'MixedTanimotoGP' object has no attribute 'input_transform'",
        ):
            assert model.model.input_transform is None
    assert model.is_compatibilized is False
    # reload the model from dump and check for equality in predictions
    model2 = MixedTanimotoGPSurrogate(
        inputs=inputs,
        outputs=outputs,
        molecular_kernel=molecular_kernel,
        continuous_kernel=continuous_kernel,
        input_preprocessing_specs=specs,
    )
    model2 = surrogates.map(model2)
    model2.loads(dump)
    preds2 = model2.predict(experiments.iloc[:-1])
    assert_frame_equal(preds, preds2)
