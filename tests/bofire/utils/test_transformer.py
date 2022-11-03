import random

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.utils._testing import assert_array_equal

from bofire.domain.domain import Domain, DomainError
from bofire.domain.features import (
    CategoricalDescriptorInputFeature,
    CategoricalInputFeature,
    ContinuousInputFeature,
    ContinuousOutputFeature,
    InputFeature,
)
from bofire.utils.transformer import (
    CategoricalEncodingEnum,
    DescriptorEncodingEnum,
    ScalerEnum,
    Transformer,
)
from tests.bofire.domain.test_domain_validators import generate_experiments
from tests.bofire.domain.test_features import (
    VALID_CATEGORICAL_DESCRIPTOR_INPUT_FEATURE_SPEC,
    VALID_CATEGORICAL_INPUT_FEATURE_SPEC,
    VALID_CONTINUOUS_INPUT_FEATURE_SPEC,
    VALID_CONTINUOUS_OUTPUT_FEATURE_SPEC,
)

if1 = ContinuousInputFeature(
    **{
        **VALID_CONTINUOUS_INPUT_FEATURE_SPEC,
        "key": "if1",
    }
)
if2 = CategoricalDescriptorInputFeature(
    **{
        **VALID_CATEGORICAL_DESCRIPTOR_INPUT_FEATURE_SPEC,
        "key": "if2",
    }
)
if3 = CategoricalInputFeature(
    **{
        **VALID_CATEGORICAL_INPUT_FEATURE_SPEC,
        "key": "if3",
    }
)
of1 = ContinuousOutputFeature(
    **{
        **VALID_CONTINUOUS_OUTPUT_FEATURE_SPEC,
        "key": "of1",
    }
)


domain = Domain(
    input_features=[if1, if2, if3],
    output_features=[of1],
    constraints=[],
)

data = generate_experiments(domain, 5)

data_dict = {
    "if1": np.random.uniform(0, 3, 3).tolist(),
    "if2": ["c1", "c3", "c2"],
    "if3": ["c1", "c3", "c2"],
    "of1": np.random.uniform(0, 3, 3).tolist(),
}

df_data = pd.DataFrame.from_dict(data_dict)


@pytest.mark.parametrize(
    "domain, descriptor_encoding, categorical_encoding",
    [
        (
            domain,
            random.choice(list(DescriptorEncodingEnum)),
            random.choice(list(CategoricalEncodingEnum)),
        )
    ],
)
def test_transformer_init(domain, descriptor_encoding, categorical_encoding):

    transformer = Transformer(domain, descriptor_encoding, categorical_encoding)

    data_transformed = transformer.fit_transform(data)
    feature_names_from_transform = data_transformed.columns.values.tolist()

    feature_names = [
        transformer.features2transformedFeatures.get(feat, [feat])
        for feat in domain.get_feature_keys()
    ]
    feature_names = [names for sublist in feature_names for names in sublist]

    if (
        descriptor_encoding == DescriptorEncodingEnum.DESCRIPTOR
        or categorical_encoding == CategoricalEncodingEnum.ONE_HOT
    ):

        feature_dict = transformer.features2transformedFeatures
        feature_names_from_init = [
            feature_name for list in feature_dict.values() for feature_name in list
        ]

        splits = [name.split("_")[0] for name in feature_names_from_init]

        assert len(splits) > 0
        assert len(set(splits).intersection(set(domain.get_feature_keys()))) > 0
        assert (
            len(
                set(feature_names_from_init).intersection(
                    set(feature_names_from_transform)
                )
            )
            > 0
        )
        assert len(set(feature_names_from_init).intersection(set(feature_names))) > 0

    assert set(feature_names) == set(feature_names_from_transform)


@pytest.mark.parametrize(
    "domain, categorical_encoding, descriptor_encoding, scale_inputs, scale_outputs, data",
    [
        (
            domain,
            categorical_encoding,
            descriptor_encoding,
            scale_inputs,
            scale_outputs,
            data,
        )
        for categorical_encoding in [i.name for i in CategoricalEncodingEnum] + [None]
        for descriptor_encoding in [i.name for i in DescriptorEncodingEnum] + [None]
        for scale_inputs in [i.name for i in ScalerEnum] + [None]
        for scale_outputs in [i.name for i in ScalerEnum] + [None]
    ],
)
def test_transformer_transform(
    domain: Domain,
    categorical_encoding: CategoricalEncodingEnum,
    descriptor_encoding: DescriptorEncodingEnum,
    scale_inputs: ScalerEnum,
    scale_outputs: ScalerEnum,
    data: pd.DataFrame,
):
    transformer = Transformer(
        domain=domain,
        descriptor_encoding=descriptor_encoding,
        categorical_encoding=categorical_encoding,
        scale_inputs=scale_inputs,
        scale_outputs=scale_outputs,
    )

    data_transformed = transformer.fit_transform(data)
    data_untransformed = transformer.inverse_transform(data_transformed)
    assert_frame_equal(data, data_untransformed)
    assert_array_equal(data_transformed, transformer.fit(data).transform(data))


@pytest.mark.parametrize(
    "domain, scale_inputs",
    [(domain, scale_inputs) for scale_inputs in [i.name for i in ScalerEnum]],
)
def test_transformer_scaling(
    domain: Domain,
    scale_inputs: ScalerEnum,
):
    transformer = Transformer(domain=domain, scale_inputs=scale_inputs)
    X = pd.DataFrame([1, 2, 3], columns=["x1"])

    var_min = min(X["x1"])
    var_max = max(X["x1"])
    transformer.fit_scaling(X.columns[0], X, var_min, var_max, scaler_type=scale_inputs)

    scaled_data = X.copy()
    if transformer.encoders["x1"] is not None:
        scaled_data = transformer.encoders["x1"].transform(X)

    scaled_X = pd.DataFrame(scaled_data, columns=X.columns)
    unscaled_X = transformer.un_scale(X.columns[0], scaled_X)

    assert all(X == unscaled_X)


@pytest.mark.parametrize("domain, scale_inputs", [(domain, "NORMALIZE")])
def test_transformer_normalize_without_extreme_values(
    domain: Domain,
    scale_inputs: ScalerEnum,
):
    transformer = Transformer(domain=domain, scale_inputs=scale_inputs)
    X = pd.DataFrame([1, 2, 3], columns=["x1"])

    transformer.fit_scaling(X.columns[0], X, scaler_type=scale_inputs)
    scaled_data = X.copy()
    if transformer.encoders["x1"] is not None:
        scaled_data = transformer.encoders["x1"].transform(X)

    scaled_X = pd.DataFrame(scaled_data, columns=X.columns)
    unscaled_X = transformer.un_scale(X.columns[0], scaled_X)

    assert all(X == unscaled_X)


@pytest.mark.parametrize(
    "domain, descriptor_encoding, categorical_encoding, exp",
    [
        (domain, descriptor_encoding, categorical_encoding, exp)
        for descriptor_encoding, categorical_encoding, exp in [
            ("DESCRIPTOR", "ONE_HOT", 3),
            ("DESCRIPTOR", "ORDINAL", 3),
            ("DESCRIPTOR", None, 2),
            ("CATEGORICAL", "ONE_HOT", 3),
            ("CATEGORICAL", "ORDINAL", 3),
            ("CATEGORICAL", None, 2),
            (None, "ONE_HOT", 2),
            (None, "ORDINAL", 2),
            (None, None, 1),
        ]
    ],
)
def test_transformer_get_features_to_be_transformed(
    domain: Domain,
    descriptor_encoding: DescriptorEncodingEnum,
    categorical_encoding: CategoricalEncodingEnum,
    exp: int,
):
    transformer = Transformer(
        domain=domain,
        categorical_encoding=categorical_encoding,
        descriptor_encoding=descriptor_encoding,
    )
    features = transformer.get_features_to_be_transformed()

    assert len(features) == exp


@pytest.mark.parametrize("domain", [domain])
def test_transformer_not_initialized(domain: Domain):
    x = pd.DataFrame([1, 2, 3])
    transformer = Transformer(domain=domain)

    with pytest.raises(AssertionError):
        transformer.transform(x)

    with pytest.raises(AssertionError):
        transformer.inverse_transform(x)


@pytest.mark.parametrize(
    "domain, df_data, descriptor_encoding, categorical_encoding, exp_dtype",
    [
        (domain, df_data, descriptor_encoding, categorical_encoding, exp_dtype)
        for descriptor_encoding, categorical_encoding, exp_dtype in [
            (
                "DESCRIPTOR",
                "ONE_HOT",
                [
                    np.dtype("float64"),
                    np.dtype("float64"),
                    np.dtype("float64"),
                    np.dtype("float64"),
                    np.dtype("float64"),
                    np.dtype("float64"),
                    np.dtype("float64"),
                ],
            ),
            (
                "DESCRIPTOR",
                "ORDINAL",
                [
                    np.dtype("float64"),
                    np.dtype("float64"),
                    np.dtype("float64"),
                    np.dtype("int32"),
                    np.dtype("float64"),
                ],
            ),
            (
                "DESCRIPTOR",
                None,
                [
                    np.dtype("float64"),
                    np.dtype("float64"),
                    np.dtype("float64"),
                    np.dtype("O"),
                    np.dtype("float64"),
                ],
            ),
            (
                "CATEGORICAL",
                "ONE_HOT",
                [
                    np.dtype("float64"),
                    np.dtype("float64"),
                    np.dtype("float64"),
                    np.dtype("float64"),
                    np.dtype("float64"),
                    np.dtype("float64"),
                    np.dtype("float64"),
                    np.dtype("float64"),
                ],
            ),
            (
                "CATEGORICAL",
                "ORDINAL",
                [
                    np.dtype("float64"),
                    np.dtype("int32"),
                    np.dtype("int32"),
                    np.dtype("float64"),
                ],
            ),
            (
                "CATEGORICAL",
                None,
                [
                    np.dtype("float64"),
                    np.dtype("O"),
                    np.dtype("O"),
                    np.dtype("float64"),
                ],
            ),
            (
                None,
                "ONE_HOT",
                [
                    np.dtype("float64"),
                    np.dtype("O"),
                    np.dtype("float64"),
                    np.dtype("float64"),
                    np.dtype("float64"),
                    np.dtype("float64"),
                ],
            ),
            (
                None,
                "ORDINAL",
                [
                    np.dtype("float64"),
                    np.dtype("O"),
                    np.dtype("int32"),
                    np.dtype("float64"),
                ],
            ),
            (
                None,
                None,
                [
                    np.dtype("float64"),
                    np.dtype("O"),
                    np.dtype("O"),
                    np.dtype("float64"),
                ],
            ),
        ]
    ],
)
def test_transformer_dtype(
    domain: Domain,
    categorical_encoding: CategoricalEncodingEnum,
    descriptor_encoding: DescriptorEncodingEnum,
    df_data: pd.DataFrame,
    exp_dtype: list,
):
    transformer = Transformer(
        domain=domain,
        categorical_encoding=categorical_encoding,
        descriptor_encoding=descriptor_encoding,
    )

    data_tr = transformer.fit_transform(df_data)
    dtypes = data_tr.dtypes

    assert_array_equal(dtypes.values, np.array(exp_dtype))
