import importlib

import pandas as pd
import pytest
import torch
from botorch.models.transforms.input import InputStandardize, Normalize

from bofire.data_models.domain.api import Inputs
from bofire.data_models.enum import CategoricalEncodingEnum
from bofire.data_models.features.api import (
    CategoricalDescriptorInput,
    CategoricalInput,
    ContinuousInput,
    MolecularInput,
)
from bofire.data_models.molfeatures.api import (
    Fingerprints,
    FingerprintsFragments,
    Fragments,
    MordredDescriptors,
)
from bofire.data_models.surrogates.api import ScalerEnum
from bofire.surrogates.utils import (
    get_categorical_feature_keys,
    get_continuous_feature_keys,
    get_molecular_feature_keys,
    get_scaler,
)
from bofire.utils.torch_tools import tkwargs


RDKIT_AVAILABLE = importlib.util.find_spec("rdkit") is not None


def test_get_scaler_none():
    inputs = Inputs(
        features=[
            CategoricalInput(key="x_cat", categories=["mama", "papa"]),
            CategoricalInput(key="x_desc", categories=["alpha", "beta"]),
        ],
    )
    scaler = get_scaler(
        inputs=inputs,
        input_preprocessing_specs={
            "x_cat": CategoricalEncodingEnum.ONE_HOT,
            "x_desc": CategoricalEncodingEnum.ONE_HOT,
        },
        scaler=ScalerEnum.NORMALIZE,
        X=inputs.sample(n=10),
    )
    assert scaler is None


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
                key=f"x_{i + 1}",
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
        ],
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
    elif scaler is None:
        with pytest.raises(AttributeError):
            assert (scaler.offset == expected_offset).all()
        with pytest.raises(AttributeError):
            assert (scaler.coefficient == expected_coefficient).all()


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
@pytest.mark.parametrize(
    "scaler_enum, input_preprocessing_specs, expected_scaler, expected_indices",
    [
        (
            ScalerEnum.NORMALIZE,
            {
                "x_mol": MordredDescriptors(descriptors=["NssCH2", "ATSC2d"]),
            },
            Normalize,
            torch.tensor([0, 1, 2, 3], dtype=torch.int64),
        ),
        (
            ScalerEnum.NORMALIZE,
            {
                "x_mol": Fingerprints(n_bits=2),
            },
            Normalize,
            torch.tensor([0, 1], dtype=torch.int64),
        ),
        (
            ScalerEnum.STANDARDIZE,
            {
                "x_mol": MordredDescriptors(descriptors=["NssCH2", "ATSC2d"]),
            },
            InputStandardize,
            torch.tensor([0, 1, 2, 3], dtype=torch.int64),
        ),
        (
            ScalerEnum.STANDARDIZE,
            {
                "x_mol": Fragments(fragments=["fr_unbrch_alkane", "fr_thiocyan"]),
            },
            InputStandardize,
            torch.tensor([0, 1], dtype=torch.int64),
        ),
        (
            ScalerEnum.IDENTITY,
            {
                "x_mol": MordredDescriptors(descriptors=["NssCH2", "ATSC2d"]),
            },
            type(None),
            None,
        ),
        (
            ScalerEnum.IDENTITY,
            {
                "x_mol": FingerprintsFragments(n_bits=32),
            },
            type(None),
            None,
        ),
    ],
)
def test_get_scaler_molecular(
    scaler_enum,
    input_preprocessing_specs,
    expected_scaler,
    expected_indices,
):
    inputs = Inputs(
        features=[
            ContinuousInput(
                key=f"x_{i + 1}",
                bounds=(0, 5),
            )
            for i in range(2)
        ]
        + [MolecularInput(key="x_mol")],
    )
    experiments = [
        [5.0, 2.5, "CC(=O)Oc1ccccc1C(=O)O"],
        [4.0, 2.0, "c1ccccc1"],
        [3.0, 0.5, "[CH3][CH2][OH]"],
        [1.5, 4.5, "N[C@](C)(F)C(=O)O"],
    ]
    experiments = pd.DataFrame(experiments, columns=["x_1", "x_2", "x_mol"])
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
        with pytest.raises(
            AttributeError,
            match="'NoneType' object has no attribute 'indices'",
        ):
            assert (scaler.indices == expected_indices).all()


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
@pytest.mark.parametrize(
    "specs, expected_continuous_keys, expected_categorical_keys, expected_molecular_keys",
    [
        (
            {
                "x2": CategoricalEncodingEnum.ONE_HOT,
                "x3": CategoricalEncodingEnum.ONE_HOT,
                "x4": Fingerprints(n_bits=2),
            },
            ["x1"],
            ["x2", "x3"],
            ["x4"],
        ),
        (
            {
                "x2": CategoricalEncodingEnum.ONE_HOT,
                "x3": CategoricalEncodingEnum.ONE_HOT,
                "x4": Fragments(fragments=["fr_unbrch_alkane", "fr_thiocyan"]),
            },
            ["x1"],
            ["x2", "x3"],
            ["x4"],
        ),
        (
            {
                "x2": CategoricalEncodingEnum.ONE_HOT,
                "x3": CategoricalEncodingEnum.ONE_HOT,
                "x4": MordredDescriptors(descriptors=["NssCH2", "ATSC2d"]),
            },
            ["x1", "x4"],
            ["x2", "x3"],
            [],
        ),
        (
            {
                "x2": CategoricalEncodingEnum.ONE_HOT,
                "x3": CategoricalEncodingEnum.DESCRIPTOR,
                "x4": Fingerprints(n_bits=2),
            },
            ["x1", "x3"],
            ["x2"],
            ["x4"],
        ),
        (
            {
                "x2": CategoricalEncodingEnum.ONE_HOT,
                "x3": CategoricalEncodingEnum.DESCRIPTOR,
                "x4": Fragments(fragments=["fr_unbrch_alkane", "fr_thiocyan"]),
            },
            ["x1", "x3"],
            ["x2"],
            ["x4"],
        ),
        (
            {
                "x2": CategoricalEncodingEnum.ONE_HOT,
                "x3": CategoricalEncodingEnum.DESCRIPTOR,
                "x4": MordredDescriptors(descriptors=["NssCH2", "ATSC2d"]),
            },
            ["x1", "x3", "x4"],
            ["x2"],
            [],
        ),
    ],
)
def test_get_feature_keys(
    specs,
    expected_continuous_keys,
    expected_categorical_keys,
    expected_molecular_keys,
):
    inps = Inputs(
        features=[
            ContinuousInput(key="x1", bounds=(0, 1)),
            CategoricalInput(key="x2", categories=["apple", "banana", "orange"]),
            CategoricalDescriptorInput(
                key="x3",
                categories=["apple", "banana", "orange", "cherry"],
                descriptors=["d1", "d2"],
                values=[[1, 2], [3, 4], [5, 6], [7, 8]],
            ),
            MolecularInput(key="x4"),
        ],
    )
    molecular_feature_keys = get_molecular_feature_keys(specs)
    continuous_feature_keys = get_continuous_feature_keys(inps, specs)
    categorical_feature_keys = get_categorical_feature_keys(specs)

    assert molecular_feature_keys == expected_molecular_keys
    assert continuous_feature_keys == expected_continuous_keys
    assert categorical_feature_keys == expected_categorical_keys
