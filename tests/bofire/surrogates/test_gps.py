import pandas as pd
import pytest
import torch
from botorch.models import MixedSingleTaskGP, SingleTaskGP
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
from bofire.data_models.enum import CategoricalEncodingEnum, MolecularEncodingEnum
from bofire.data_models.features.api import (
    CategoricalDescriptorInput,
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
    MolecularInput,
)
from bofire.data_models.kernels.api import (
    HammondDistanceKernel,
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
from bofire.data_models.surrogates.api import (
    MixedSingleTaskGPSurrogate,
    ScalerEnum,
    SingleTaskGPSurrogate,
)
from bofire.surrogates.single_task_gp import get_scaler
from bofire.utils.torch_tools import tkwargs

RDKIT_AVAILABLE = importlib.util.find_spec("rdkit") is not None

@pytest.mark.parametrize(
    "scaler_enum, input_preprocessing_specs, expected_scaler, expected_indices, expected_offset, expected_coefficient",
    [
        (
            ScalerEnum.NORMALIZE,
            {
                "x_cat": CategoricalEncodingEnum.ONE_HOT,
                "x_desc": CategoricalEncodingEnum.ONE_HOT,
            },
            Normalize,
            torch.tensor([0, 1], dtype=torch.int64),
            torch.tensor([-4.0, -4.0]).to(**tkwargs),
            torch.tensor([8.0, 8.0]).to(**tkwargs),
        ),
        (
            ScalerEnum.NORMALIZE,
            {
                "x_cat": CategoricalEncodingEnum.ONE_HOT,
                "x_desc": CategoricalEncodingEnum.DESCRIPTOR,
            },
            Normalize,
            torch.tensor([0, 1, 2], dtype=torch.int64),
            torch.tensor([-4.0, -4.0, 1.0]).to(**tkwargs),
            torch.tensor([8.0, 8.0, 5.0]).to(**tkwargs),
        ),
        (
            ScalerEnum.STANDARDIZE,
            {
                "x_cat": CategoricalEncodingEnum.ONE_HOT,
                "x_desc": CategoricalEncodingEnum.ONE_HOT,
            },
            InputStandardize,
            torch.tensor([0, 1], dtype=torch.int64),
            None,
            None,
        ),
        (
            ScalerEnum.STANDARDIZE,
            {
                "x_cat": CategoricalEncodingEnum.ONE_HOT,
                "x_desc": CategoricalEncodingEnum.DESCRIPTOR,
            },
            InputStandardize,
            torch.tensor([0, 1, 2], dtype=torch.int64),
            None,
            None,
        ),
        (
            ScalerEnum.IDENTITY,
            {
                "x_cat": CategoricalEncodingEnum.ONE_HOT,
                "x_desc": CategoricalEncodingEnum.ONE_HOT,
            },
            type(None),
            None,
            None,
            None,
        ),
        (
            ScalerEnum.IDENTITY,
            {
                "x_cat": CategoricalEncodingEnum.ONE_HOT,
                "x_desc": CategoricalEncodingEnum.DESCRIPTOR,
            },
            type(None),
            None,
            None,
            None,
        ),
    ],
)
def test_get_scaler(
    scaler_enum,
    input_preprocessing_specs,
    expected_scaler,
    expected_indices,
    expected_offset,
    expected_coefficient,
):
    inputs = Inputs(
        features=[
            ContinuousInput(
                key=f"x_{i+1}",
                bounds=(-4, 4),
            )
            for i in range(2)
        ]
        + [
            CategoricalInput(key="x_cat", categories=["mama", "papa"]),
            CategoricalDescriptorInput(
                key="x_desc",
                categories=["alpha", "beta"],
                descriptors=["oskar"],
                values=[[1], [6]],
            ),
        ]
    )
    experiments = inputs.sample(n=10)
    scaler = get_scaler(
        inputs=inputs,
        input_preprocessing_specs=input_preprocessing_specs,
        scaler=scaler_enum,
        X=experiments[inputs.get_keys()],
    )
    assert isinstance(scaler, expected_scaler)
    if expected_indices is not None:
        assert (scaler.indices == expected_indices).all()
    else:
        with pytest.raises(AttributeError):
            assert (scaler.indices == expected_indices).all()
    if expected_offset is not None:
        assert torch.allclose(scaler.offset, expected_offset)
        assert torch.allclose(scaler.coefficient, expected_coefficient)
    else:
        if scaler is None:
            with pytest.raises(AttributeError):
                assert (scaler.offset == expected_offset).all()
            with pytest.raises(AttributeError):
                assert (scaler.coefficient == expected_coefficient).all()


@pytest.mark.parametrize(
    "kernel, scaler",
    [
        (ScaleKernel(base_kernel=RBFKernel(ard=True)), ScalerEnum.NORMALIZE),
        (ScaleKernel(base_kernel=RBFKernel(ard=False)), ScalerEnum.STANDARDIZE),
        (ScaleKernel(base_kernel=RBFKernel(ard=False)), ScalerEnum.IDENTITY),
    ],
)
def test_SingleTaskGPModel(kernel, scaler):
    inputs = Inputs(
        features=[
            ContinuousInput(
                key=f"x_{i+1}",
                bounds=(-4, 4),
            )
            for i in range(2)
        ]
    )
    outputs = Outputs(features=[ContinuousOutput(key="y")])
    experiments = inputs.sample(n=10)
    experiments.eval("y=((x_1**2 + x_2 - 11)**2+(x_1 + x_2**2 -7)**2)", inplace=True)
    experiments["valid_y"] = 1
    model = SingleTaskGPSurrogate(
        inputs=inputs,
        outputs=outputs,
        kernel=kernel,
        scaler=scaler,
    )
    model = surrogates.map(model)
    samples = inputs.sample(5)
    # test error on non fitted model
    with pytest.raises(ValueError):
        model.predict(samples)
    model.fit(experiments)
    # dump the model
    dump = model.dumps()
    # make predictions
    samples2 = samples.copy()
    samples2 = samples2.astype({"x_1": "object"})
    preds = model.predict(samples2)
    assert preds.shape == (5, 2)
    # check that model is composed correctly
    assert isinstance(model.model, SingleTaskGP)
    assert isinstance(model.model.outcome_transform, Standardize)
    if scaler == ScalerEnum.NORMALIZE:
        assert isinstance(model.model.input_transform, Normalize)
    elif scaler == ScalerEnum.STANDARDIZE:
        assert isinstance(model.model.input_transform, InputStandardize)
    else:
        with pytest.raises(AttributeError):
            assert model.model.input_transform is None
    assert model.is_compatibilized is False
    # reload the model from dump and check for equality in predictions
    model2 = SingleTaskGPSurrogate(
        inputs=inputs,
        outputs=outputs,
        kernel=kernel,
        scaler=scaler,
    )
    model2 = surrogates.map(model2)
    model2.loads(dump)
    preds2 = model2.predict(samples)
    assert_frame_equal(preds, preds2)


smiles = [
    "CC(=O)Oc1ccccc1C(=O)O",
    "c1ccccc1",
    "[CH3][CH2][OH]",
    "N[C@](C)(F)C(=O)O",
]

@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
@pytest.mark.parametrize(
    "input_feature, kernel, scaler, specs",
    [
        (
            [
                MolecularInput(
                    key="x_1", smiles=smiles, molfeatures=Fingerprints(n_bits=32)
                )
            ],
            ScaleKernel(base_kernel=TanimotoKernel(ard=False)),
            ScalerEnum.IDENTITY,
            {"x_1": MolecularEncodingEnum.FINGERPRINTS},
        ),
        (
            [MolecularInput(key="x_1", smiles=smiles, molfeatures=Fragments())],
            ScaleKernel(base_kernel=TanimotoKernel(ard=False)),
            ScalerEnum.IDENTITY,
            {"x_1": MolecularEncodingEnum.FRAGMENTS},
        ),
        (
            [
                MolecularInput(
                    key="x_1",
                    smiles=smiles,
                    molfeatures=FingerprintsFragments(n_bits=32),
                )
            ],
            ScaleKernel(base_kernel=TanimotoKernel(ard=False)),
            ScalerEnum.IDENTITY,
            {"x_1": MolecularEncodingEnum.FINGERPRINTS_FRAGMENTS},
        ),
        (
            [
                MolecularInput(
                    key="x_1",
                    smiles=smiles,
                    molfeatures=MordredDescriptors(descriptors=["NssCH2", "ATSC2d"]),
                )
            ],
            ScaleKernel(base_kernel=RBFKernel(ard=False)),
            ScalerEnum.NORMALIZE,
            {"x_1": MolecularEncodingEnum.MOL_DESCRIPTOR},
        ),
    ],
)
def test_SingleTaskGPModel_molecular(input_feature, kernel, scaler, specs):
    inputs = Inputs(features=input_feature)
    outputs = Outputs(features=[ContinuousOutput(key="y")])
    experiments = [
        ["CC(=O)Oc1ccccc1C(=O)O", 88.0],
        ["c1ccccc1", 35.0],
        ["[CH3][CH2][OH]", 69.0],
        ["N[C@](C)(F)C(=O)O", 20.0],
    ]
    experiments = pd.DataFrame(experiments, columns=["x_1", "y"])
    experiments["valid_y"] = 1
    model = SingleTaskGPSurrogate(
        inputs=inputs,
        outputs=outputs,
        kernel=kernel,
        scaler=scaler,
        input_preprocessing_specs=specs,
    )
    model = surrogates.map(model)
    samples = inputs.sample(5)
    # test error on non fitted model
    with pytest.raises(ValueError):
        model.predict(samples)
    model.fit(experiments)
    # dump the model
    dump = model.dumps()
    # make predictions
    samples2 = samples.copy()
    samples2 = samples2.astype({"x_1": "object"})
    preds = model.predict(samples2)
    assert preds.shape == (5, 2)
    # check that model is composed correctly
    assert isinstance(model.model, SingleTaskGP)
    assert isinstance(model.model.outcome_transform, Standardize)
    if scaler == ScalerEnum.NORMALIZE:
        assert isinstance(model.model.input_transform, Normalize)
    elif scaler == ScalerEnum.STANDARDIZE:
        assert isinstance(model.model.input_transform, InputStandardize)
    else:
        with pytest.raises(AttributeError):
            assert model.model.input_transform is None
    assert model.is_compatibilized is False
    # reload the model from dump and check for equality in predictions
    model2 = SingleTaskGPSurrogate(
        inputs=inputs,
        outputs=outputs,
        kernel=kernel,
        scaler=scaler,
        input_preprocessing_specs=specs,
    )
    model2 = surrogates.map(model2)
    model2.loads(dump)
    preds2 = model2.predict(samples)
    assert_frame_equal(preds, preds2)


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
@pytest.mark.parametrize(
    "input_feature, scaler, specs",
    [
        (
            [
                MolecularInput(
                    key="x_1", smiles=smiles, molfeatures=Fingerprints(n_bits=32)
                )
            ],
            ScalerEnum.IDENTITY,
            {"x_1": MolecularEncodingEnum.FRAGMENTS},
        ),
        (
            [MolecularInput(key="x_1", smiles=smiles, molfeatures=Fragments())],
            ScalerEnum.IDENTITY,
            {"x_1": MolecularEncodingEnum.FINGERPRINTS},
        ),
        (
            [
                MolecularInput(
                    key="x_1",
                    smiles=smiles,
                    molfeatures=FingerprintsFragments(n_bits=32),
                )
            ],
            ScalerEnum.IDENTITY,
            {"x_1": MolecularEncodingEnum.MOL_DESCRIPTOR},
        ),
        (
            [
                MolecularInput(
                    key="x_1",
                    smiles=smiles,
                    molfeatures=MordredDescriptors(descriptors=["NssCH2", "ATSC2d"]),
                )
            ],
            ScalerEnum.NORMALIZE,
            {"x_1": MolecularEncodingEnum.FINGERPRINTS_FRAGMENTS},
        ),
    ],
)
def test_SingleTaskGPModel_molecular_invalid_input_preprocessing_specs(
    input_feature, scaler, specs
):
    inputs = Inputs(features=input_feature)
    outputs = Outputs(features=[ContinuousOutput(key="y")])
    with pytest.raises(ValueError):
        SingleTaskGPSurrogate(
            inputs=inputs,
            outputs=outputs,
            kernel=ScaleKernel(base_kernel=TanimotoKernel(ard=False)),
            scaler=scaler,
            input_preprocessing_specs=specs,
        )


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
@pytest.mark.parametrize(
    "molfeatures, kernel, specs",
    [
        (
            Fingerprints(n_bits=32),
            ScaleKernel(base_kernel=RBFKernel(ard=True)),
            {"x_1": MolecularEncodingEnum.FINGERPRINTS},
        ),
        (
            Fragments(),
            ScaleKernel(base_kernel=MaternKernel(ard=True)),
            {"x_1": MolecularEncodingEnum.FRAGMENTS},
        ),
        (
            FingerprintsFragments(n_bits=32),
            ScaleKernel(base_kernel=RBFKernel(ard=True)),
            {"x_1": MolecularEncodingEnum.FINGERPRINTS_FRAGMENTS},
        ),
    ],
)
def test_SingleTaskGPModel_molecular_invalid_kernel(molfeatures, kernel, specs):
    inputs = Inputs(
        features=[MolecularInput(key="x_1", smiles=smiles, molfeatures=molfeatures)]
    )
    outputs = Outputs(features=[ContinuousOutput(key="y")])
    with pytest.raises(ValueError):
        SingleTaskGPSurrogate(
            inputs=inputs,
            outputs=outputs,
            kernel=kernel,
            scaler=ScalerEnum.IDENTITY,
            input_preprocessing_specs=specs,
        )


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
@pytest.mark.parametrize(
    "molfeatures, scaler, specs",
    [
        (
            Fingerprints(n_bits=32),
            ScalerEnum.NORMALIZE,
            {"x_1": MolecularEncodingEnum.FINGERPRINTS},
        ),
        (Fragments(), ScalerEnum.STANDARDIZE, {"x_1": MolecularEncodingEnum.FRAGMENTS}),
        (
            FingerprintsFragments(n_bits=32),
            ScalerEnum.NORMALIZE,
            {"x_1": MolecularEncodingEnum.FINGERPRINTS_FRAGMENTS},
        ),
        (
            MordredDescriptors(descriptors=["NssCH2", "ATSC2d"]),
            ScalerEnum.STANDARDIZE,
            {"x_1": MolecularEncodingEnum.MOL_DESCRIPTOR},
        ),
    ],
)
def test_SingleTaskGPModel_tanimoto_invalid_scaler(molfeatures, scaler, specs):
    inputs = Inputs(
        features=[MolecularInput(key="x_1", smiles=smiles, molfeatures=molfeatures)]
    )
    outputs = Outputs(features=[ContinuousOutput(key="y")])
    with pytest.raises(ValueError):
        SingleTaskGPSurrogate(
            inputs=inputs,
            outputs=outputs,
            kernel=ScaleKernel(base_kernel=TanimotoKernel(ard=False)),
            scaler=scaler,
            input_preprocessing_specs=specs,
        )


def test_MixedGPModel_invalid_preprocessing():
    inputs = Inputs(
        features=[
            ContinuousInput(
                key=f"x_{i+1}",
                bounds=(-4, 4),
            )
            for i in range(2)
        ]
    )
    outputs = Outputs(features=[ContinuousOutput(key="y")])
    experiments = inputs.sample(n=10)
    experiments.eval("y=((x_1**2 + x_2 - 11)**2+(x_1 + x_2**2 -7)**2)", inplace=True)
    experiments["valid_y"] = 1
    with pytest.raises(ValidationError):
        MixedSingleTaskGPSurrogate(
            inputs=inputs,
            outputs=outputs,
        )


@pytest.mark.parametrize(
    "kernel, scaler",
    [
        (RBFKernel(ard=True), ScalerEnum.NORMALIZE),
        (RBFKernel(ard=False), ScalerEnum.STANDARDIZE),
        (RBFKernel(ard=False), ScalerEnum.IDENTITY),
    ],
)
def test_MixedGPModel(kernel, scaler):
    inputs = Inputs(
        features=[
            ContinuousInput(
                key=f"x_{i+1}",
                bounds=(-4, 4),
            )
            for i in range(2)
        ]
        + [CategoricalInput(key="x_cat", categories=["mama", "papa"])]
    )
    outputs = Outputs(features=[ContinuousOutput(key="y")])
    experiments = inputs.sample(n=10)
    experiments.eval("y=((x_1**2 + x_2 - 11)**2+(x_1 + x_2**2 -7)**2)", inplace=True)
    experiments.loc[experiments.x_cat == "mama", "y"] *= 5.0
    experiments.loc[experiments.x_cat == "papa", "y"] /= 2.0
    experiments["valid_y"] = 1

    model = MixedSingleTaskGPSurrogate(
        inputs=inputs,
        outputs=outputs,
        input_preprocessing_specs={"x_cat": CategoricalEncodingEnum.ONE_HOT},
        scaler=scaler,
        continuous_kernel=kernel,
        categorical_kernel=HammondDistanceKernel(),
    )
    model = surrogates.map(model)
    with pytest.raises(ValueError):
        model.dumps()
    model.fit(experiments)
    # dump the model
    dump = model.dumps()
    # make predictions
    samples = inputs.sample(5)
    preds = model.predict(samples)
    assert preds.shape == (5, 2)
    # check that model is composed correctly
    assert isinstance(model.model, MixedSingleTaskGP)
    assert isinstance(model.model.outcome_transform, Standardize)
    if scaler == ScalerEnum.NORMALIZE:
        assert isinstance(model.model.input_transform, ChainedInputTransform)
        assert isinstance(model.model.input_transform.tf1, Normalize)
        assert torch.eq(
            model.model.input_transform.tf1.indices,
            torch.tensor([0, 1], dtype=torch.int64),
        ).all()
        assert isinstance(model.model.input_transform.tf2, OneHotToNumeric)
    elif scaler == ScalerEnum.STANDARDIZE:
        assert isinstance(model.model.input_transform, ChainedInputTransform)
        assert isinstance(model.model.input_transform.tf1, InputStandardize)
        assert torch.eq(
            model.model.input_transform.tf1.indices,
            torch.tensor([0, 1], dtype=torch.int64),
        ).all()
        assert isinstance(model.model.input_transform.tf2, OneHotToNumeric)
    else:
        assert isinstance(model.model.input_transform, OneHotToNumeric)
    assert model.is_compatibilized is False
    # reload the model from dump and check for equality in predictions
    model2 = MixedSingleTaskGPSurrogate(
        inputs=inputs,
        outputs=outputs,
        continuous_kernel=kernel,
        scaler=scaler,
    )
    model2 = surrogates.map(model2)
    model2.loads(dump)
    preds2 = model2.predict(samples)
    assert_frame_equal(preds, preds2)
