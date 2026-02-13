import importlib

import pandas as pd
import pytest
import torch
from botorch.models.transforms.input import (
    ChainedInputTransform,
    FilterFeatures,
    InputStandardize,
    Normalize,
    NumericToCategoricalEncoding,
)

from bofire.data_models.domain.api import EngineeredFeatures, Inputs
from bofire.data_models.enum import CategoricalEncodingEnum
from bofire.data_models.features.api import (
    CategoricalDescriptorInput,
    CategoricalInput,
    CategoricalMolecularInput,
    ContinuousDescriptorInput,
    ContinuousInput,
    MeanFeature,
    SumFeature,
    WeightedSumFeature,
)
from bofire.data_models.molfeatures.api import (
    CompositeMolFeatures,
    Fingerprints,
    FingerprintsFragments,
    Fragments,
    MordredDescriptors,
)
from bofire.data_models.surrogates.api import ScalerEnum
from bofire.surrogates.utils import (
    get_continuous_feature_keys,
    get_input_transform,
    get_scaler,
)


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
        engineered_features=EngineeredFeatures(features=[]),
        categorical_encodings={
            "x_cat": CategoricalEncodingEnum.ONE_HOT,
            "x_desc": CategoricalEncodingEnum.ONE_HOT,
        },
        scaler_type=ScalerEnum.NORMALIZE,
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
            None,
            None,
        ),
        (
            ScalerEnum.NORMALIZE,
            {
                "x_cat": CategoricalEncodingEnum.ONE_HOT,
                "x_desc": CategoricalEncodingEnum.DESCRIPTOR,
            },
            Normalize,
            torch.tensor([0, 1, 2, 3], dtype=torch.int64),
            None,
            None,
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
            torch.tensor([0, 1, 2, 3], dtype=torch.int64),
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
                descriptors=["oskar", "wilde"],
                values=[[1, 3], [6, 8]],
            ),
        ],
    )

    scaler_dict = get_scaler(
        inputs=inputs,
        engineered_features=EngineeredFeatures(features=[]),
        categorical_encodings=input_preprocessing_specs,
        scaler_type=scaler_enum,
    )
    scaler = None if scaler_dict is None else scaler_dict["scaler"]
    assert isinstance(scaler, expected_scaler)
    if expected_indices is not None:
        assert (scaler.indices == expected_indices).all()
        assert scaler.transform_on_train is True
    else:
        with pytest.raises(AttributeError):
            assert (scaler.indices == expected_indices).all()
    # if expected_offset is not None:
    #     assert torch.allclose(scaler.offset, expected_offset)
    #     assert torch.allclose(scaler.coefficient, expected_coefficient)
    # elif scaler is None:
    #     with pytest.raises(AttributeError):
    #         assert (scaler.offset == expected_offset).all()
    #     with pytest.raises(AttributeError):
    #         assert (scaler.coefficient == expected_coefficient).all()


def test_get_scaler_with_experiments():
    inputs = Inputs(
        features=[
            ContinuousInput(
                key=f"x_{i + 1}",
                bounds=(-4, 4),
            )
            for i in range(2)
        ]
    )

    scaler = get_scaler(
        inputs=inputs,
        engineered_features=EngineeredFeatures(features=[]),
        categorical_encodings={},
        scaler_type=ScalerEnum.NORMALIZE,
    )["scaler"]

    assert isinstance(scaler, Normalize)
    assert (scaler.bounds == torch.tensor([[-4.0], [4.0]])).all()

    experiments_beyond_bounds = pd.DataFrame(
        [[-8.0, 0.1], [1.2, 5.0]], columns=inputs.get_keys()
    )

    scaler_beyond_bounds = get_scaler(
        inputs=inputs,
        engineered_features=EngineeredFeatures(features=[]),
        categorical_encodings={},
        scaler_type=ScalerEnum.NORMALIZE,
        X=experiments_beyond_bounds,
    )["scaler"]

    assert (
        scaler_beyond_bounds.bounds == torch.tensor([[-8.0, -4.0], [4.0, 5.0]])
    ).all()


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
        + [
            CategoricalMolecularInput(
                key="x_mol",
                categories=[
                    "CC(=O)Oc1ccccc1C(=O)O",
                    "c1ccccc1",
                    "[CH3][CH2][OH]",
                    "N[C@](C)(F)C(=O)O",
                ],
            )
        ],
    )
    experiments = [
        [5.0, 2.5, "CC(=O)Oc1ccccc1C(=O)O"],
        [4.0, 2.0, "c1ccccc1"],
        [3.0, 0.5, "[CH3][CH2][OH]"],
        [1.5, 4.5, "N[C@](C)(F)C(=O)O"],
    ]
    experiments = pd.DataFrame(experiments, columns=["x_1", "x_2", "x_mol"])
    scaler_dict = get_scaler(
        inputs=inputs,
        engineered_features=EngineeredFeatures(features=[]),
        categorical_encodings=input_preprocessing_specs,
        scaler_type=scaler_enum,
        # X=experiments[inputs.get_keys()],
    )
    scaler = None if scaler_dict is None else scaler_dict["scaler"]
    assert isinstance(scaler, expected_scaler)
    if expected_indices is not None:
        assert scaler.transform_on_train is True
        assert (scaler.indices == expected_indices).all()


def test_get_scaler_engineered_features():
    inputs = Inputs(
        features=[
            ContinuousDescriptorInput(
                key=f"x_{i + 1}",
                bounds=(0, 5),
                descriptors=["d1", "d2", "d3"],
                values=[1.0, 2.0, 3.0],
            )
            for i in range(2)
        ]
        + [CategoricalInput(key="x_cat", categories=["mama", "papa", "lotta"])],
    )
    engineered_features = EngineeredFeatures(
        features=[
            SumFeature(key="sum", features=["x_1", "x_2"]),
            MeanFeature(key="mean", features=["x_1", "x_2"]),
            WeightedSumFeature(
                key="weighted_sum", features=["x_1", "x_2"], descriptors=["d1", "d3"]
            ),
        ],
    )
    scaler_dict = get_scaler(
        inputs=inputs,
        engineered_features=engineered_features,
        categorical_encodings={"x_cat": CategoricalEncodingEnum.ONE_HOT},
        scaler_type=ScalerEnum.NORMALIZE,
    )

    assert isinstance(scaler_dict, dict) and all(
        isinstance(tf, Normalize) for tf in scaler_dict.values()
    )
    assert (
        scaler_dict["scaler"].indices == torch.tensor([0, 1], dtype=torch.int64)
    ).all()
    assert not scaler_dict["scaler"].learn_coefficients

    assert (
        scaler_dict["engineered_scaler"].indices
        == torch.tensor([5, 6, 7, 8], dtype=torch.int64)
    ).all()
    assert scaler_dict["engineered_scaler"].learn_coefficients


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
@pytest.mark.parametrize(
    "specs, expected_continuous_keys",
    [
        (
            {
                "x2": CategoricalEncodingEnum.ONE_HOT,
                "x3": CategoricalEncodingEnum.ONE_HOT,
                "x4": Fingerprints(n_bits=2),
            },
            ["x1"],
        ),
        (
            {
                "x2": CategoricalEncodingEnum.ONE_HOT,
                "x3": CategoricalEncodingEnum.ONE_HOT,
                "x4": Fragments(fragments=["fr_unbrch_alkane", "fr_thiocyan"]),
            },
            ["x1"],
        ),
        (
            {
                "x2": CategoricalEncodingEnum.ONE_HOT,
                "x3": CategoricalEncodingEnum.ONE_HOT,
                "x4": MordredDescriptors(descriptors=["NssCH2", "ATSC2d"]),
            },
            ["x1", "x4"],
        ),
        (
            {
                "x2": CategoricalEncodingEnum.ONE_HOT,
                "x3": CategoricalEncodingEnum.DESCRIPTOR,
                "x4": Fingerprints(n_bits=2),
            },
            ["x1", "x3"],
        ),
        (
            {
                "x2": CategoricalEncodingEnum.ONE_HOT,
                "x3": CategoricalEncodingEnum.DESCRIPTOR,
                "x4": Fragments(fragments=["fr_unbrch_alkane", "fr_thiocyan"]),
            },
            ["x1", "x3"],
        ),
        (
            {
                "x2": CategoricalEncodingEnum.ONE_HOT,
                "x3": CategoricalEncodingEnum.DESCRIPTOR,
                "x4": FingerprintsFragments(
                    fragments=["fr_unbrch_alkane", "fr_thiocyan"], n_bits=32
                ),
            },
            ["x1", "x3"],
        ),
        (
            {
                "x2": CategoricalEncodingEnum.ONE_HOT,
                "x3": CategoricalEncodingEnum.DESCRIPTOR,
                "x4": MordredDescriptors(descriptors=["NssCH2", "ATSC2d"]),
            },
            ["x1", "x3", "x4"],
        ),
        (
            {
                "x2": CategoricalEncodingEnum.ONE_HOT,
                "x3": CategoricalEncodingEnum.DESCRIPTOR,
                "x4": CompositeMolFeatures(
                    features=[
                        MordredDescriptors(descriptors=["NssCH2", "ATSC2d"]),
                        Fingerprints(n_bits=128),
                    ]
                ),
            },
            ["x1", "x3", "x4"],
        ),
    ],
)
def test_get_feature_keys(
    specs,
    expected_continuous_keys,
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
            CategoricalMolecularInput(
                key="x4",
                categories=[
                    "CC(=O)Oc1ccccc1C(=O)O",
                    "c1ccccc1",
                    "[CH3][CH2][OH]",
                    "N[C@](C)(F)C(=O)O",
                ],
            ),
        ],
    )
    continuous_feature_keys = get_continuous_feature_keys(inps, specs)

    assert continuous_feature_keys == expected_continuous_keys


def test_get_input_transform():
    inputs = Inputs(
        features=[
            ContinuousInput(key="x1", bounds=(0, 1)),
            CategoricalInput(key="x2", categories=["apple", "banana", "orange"]),
            ContinuousInput(key="x3", bounds=(-5, 5)),
        ]
    )

    # case 1 scaler not none, categorical transform not none
    input_transform = get_input_transform(
        inputs=inputs,
        scaler_type=ScalerEnum.NORMALIZE,
        categorical_encodings={
            "x2": CategoricalEncodingEnum.ONE_HOT,
        },
        engineered_features=EngineeredFeatures(features=[]),
    )
    assert isinstance(input_transform, ChainedInputTransform)
    # case 2 scaler is none, categorical transform is not none
    input_transform = get_input_transform(
        inputs=inputs,
        scaler_type=ScalerEnum.IDENTITY,
        categorical_encodings={
            "x2": CategoricalEncodingEnum.ONE_HOT,
        },
        engineered_features=EngineeredFeatures(features=[]),
    )
    assert isinstance(input_transform, NumericToCategoricalEncoding)
    # case 3 scaler is not none, categorical transform is none
    input_transform = get_input_transform(
        inputs=inputs,
        scaler_type=ScalerEnum.NORMALIZE,
        categorical_encodings={
            "x2": CategoricalEncodingEnum.ORDINAL,
        },
        engineered_features=EngineeredFeatures(features=[]),
    )
    assert isinstance(input_transform, Normalize)
    # case 4 both is none
    input_transform = get_input_transform(
        inputs=inputs,
        scaler_type=ScalerEnum.IDENTITY,
        categorical_encodings={
            "x2": CategoricalEncodingEnum.ORDINAL,
        },
        engineered_features=EngineeredFeatures(features=[]),
    )
    assert input_transform is None
    # case 5 engineered features with scaler and categorical transform
    input_transform = get_input_transform(
        inputs=inputs,
        scaler_type=ScalerEnum.NORMALIZE,
        categorical_encodings={
            "x2": CategoricalEncodingEnum.ONE_HOT,
        },
        engineered_features=EngineeredFeatures(
            features=[
                SumFeature(key="sum", features=["x1", "x3"]),
            ],
        ),
    )
    assert isinstance(input_transform, ChainedInputTransform)
    assert list(input_transform.keys()) == ["cat", "sum", "scaler", "engineered_scaler"]
    scaler, engineered_scaler = (
        input_transform["scaler"],
        input_transform["engineered_scaler"],
    )
    assert isinstance(scaler, Normalize) and isinstance(engineered_scaler, Normalize)
    assert (scaler.indices == torch.tensor([0, 1], dtype=torch.int64)).all()
    assert (engineered_scaler.indices == torch.tensor([5], dtype=torch.int64)).all()
    # case 6 engineered features keep_features = False
    input_transform = get_input_transform(
        inputs=inputs,
        scaler_type=ScalerEnum.NORMALIZE,
        categorical_encodings={
            "x2": CategoricalEncodingEnum.ONE_HOT,
        },
        engineered_features=EngineeredFeatures(
            features=[
                SumFeature(key="sum", features=["x1", "x3"], keep_features=False),
            ],
        ),
    )
    assert isinstance(input_transform, ChainedInputTransform)
    assert list(input_transform.keys()) == [
        "cat",
        "sum",
        "scaler",
        "engineered_scaler",
        "filter_engineered",
    ]
    filter = input_transform["filter_engineered"]
    assert isinstance(filter, FilterFeatures)
    assert (
        filter.feature_indices == torch.tensor([2, 3, 4, 5], dtype=torch.int64)
    ).all()
