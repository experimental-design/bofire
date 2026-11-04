import random

import pandas as pd
import pytest
from pandas.testing import assert_series_equal

import tests.bofire.data_models.specs.api as specs
from bofire.data_models.encodings.api import (
    DescriptorEncoding,
    OneHotEncoding,
    OrdinalEncoding,
)
from bofire.data_models.features.api import CategoricalInput, CategoricalOutput
from bofire.data_models.features.descriptors import Descriptors
from bofire.data_models.objectives.api import ConstrainedCategoricalObjective


@pytest.mark.parametrize(
    "input_feature, expected",
    [
        (
            CategoricalInput(key="if1", categories=["a", "b"], allowed=[True, True]),
            ["a", "b"],
        ),
        (
            CategoricalInput(key="if2", categories=["a", "b"], allowed=[True, True]),
            ["a", "b"],
        ),
        (
            CategoricalInput(key="if3", categories=["a", "b"], allowed=[True, False]),
            ["a"],
        ),
        (
            CategoricalInput(key="if4", categories=["a", "b"], allowed=[False, True]),
            ["b"],
        ),
    ],
)
def test_categorical_input_feature_get_possible_categories(input_feature, expected):
    experiments = pd.DataFrame(
        {"if1": ["a", "b"], "if2": ["a", "a"], "if3": ["a", "a"], "if4": ["b", "b"]},
    )
    categories = input_feature.get_possible_categories(experiments[input_feature.key])
    assert categories == expected


@pytest.mark.parametrize(
    "input_feature, values, strict",
    [
        (
            specs.features.valid(CategoricalInput).obj(categories=["a", "b", "c"]),
            pd.Series([random.choice(["a", "b", "c"]) for _ in range(20)]),
            True,
        ),
        (
            specs.features.valid(CategoricalInput).obj(categories=["a", "b", "c"]),
            pd.Series([random.choice(["a", "b", "c"]) for _ in range(20)]),
            False,
        ),
        (
            specs.features.valid(CategoricalInput).obj(
                categories=["a", "b", "c"],
                allowed=[True, False, False],
            ),
            pd.Series([random.choice(["a", "b", "c"]) for _ in range(20)]),
            # CategoricalInput(**VALID_FIXED_CATEGORICAL_INPUT_FEATURE_SPEC),
            # pd.Series([random.choice(["c1", "c2", "c3"]) for _ in range(20)]),
            True,
        ),
        (
            specs.features.valid(CategoricalInput).obj(
                categories=["1", "2", "3"],
                allowed=[True, False, False],
            ),
            pd.Series([random.choice([1, 2, 3]) for _ in range(20)]),
            # CategoricalInput(**VALID_FIXED_CATEGORICAL_INPUT_FEATURE_SPEC),
            # pd.Series([random.choice(["c1", "c2", "c3"]) for _ in range(20)]),
            False,
        ),
        (
            # only one category present in the data: unused categories are an error
            # under strict (see the invalid case below), but fine without it
            specs.features.valid(CategoricalInput).obj(
                categories=["a", "b", "c"],
                allowed=[True, False, False],
            ),
            pd.Series(["a", "a"]),
            False,
        ),
    ],
)
def test_categorical_input_feature_validate_valid(input_feature, values, strict):
    input_feature.validate_experimental(values, strict)


@pytest.mark.parametrize(
    "input_feature, values, strict",
    [
        (
            specs.features.valid(CategoricalInput).obj(categories=["a", "b", "c"]),
            pd.Series(["a", "b", "c", "d"]),
            True,
        ),
        (
            specs.features.valid(CategoricalInput).obj(categories=["a", "b", "c"]),
            pd.Series(["a", "b", "c", "d"]),
            False,
        ),
        (
            specs.features.valid(CategoricalInput).obj(
                categories=["a", "b", "c"],
                allowed=[True, False, False],
            ),
            pd.Series(["a", "a"]),
            True,
        ),
        (
            specs.features.valid(CategoricalInput).obj(
                categories=["a", "b", "c"],
                allowed=[True, False, False],
            ),
            pd.Series(["a", "b"]),
            True,
        ),
        (
            specs.features.valid(CategoricalInput).obj(
                categories=["1", "2", "3"],
                allowed=[True, False, False],
            ),
            pd.Series([1, 2]),
            # CategoricalInput(**VALID_FIXED_CATEGORICAL_INPUT_FEATURE_SPEC),
            # pd.Series([random.choice(["c1", "c2", "c3"]) for _ in range(20)]),
            True,
        ),
        (
            specs.features.valid(CategoricalInput).obj(
                categories=["one", "two", "three"],
                allowed=[True, False, False],
            ),
            pd.Series([1, 2, 3]),
            # CategoricalInput(**VALID_FIXED_CATEGORICAL_INPUT_FEATURE_SPEC),
            # pd.Series([random.choice(["c1", "c2", "c3"]) for _ in range(20)]),
            True,
        ),
    ],
)
def test_categorical_input_feature_validate_invalid(input_feature, values, strict):
    with pytest.raises(ValueError):
        input_feature.validate_experimental(values, strict)


@pytest.mark.parametrize(
    "input_feature, values",
    [
        (
            specs.features.valid(CategoricalInput).obj(
                categories=["c1", "c2", "c3"],
                allowed=[True, True, True],
            ),
            pd.Series([random.choice(["c1", "c2", "c3"]) for _ in range(20)]),
        ),
        (
            specs.features.valid(CategoricalInput).obj(
                categories=["a", "b"],
                allowed=[True, False],
            ),
            pd.Series(["a", "a"]),
        ),
    ],
)
def test_categorical_input_feature_validate_candidental_valid(input_feature, values):
    input_feature.validate_candidental(values)


@pytest.mark.parametrize(
    "input_feature, values",
    [
        (
            specs.features.valid(CategoricalInput).obj(categories=["a", "b", "c"]),
            pd.Series(["a", "b", "c", "d"]),
        ),
        (
            specs.features.valid(CategoricalInput).obj(
                categories=["a", "b"],
                allowed=[True, False],
            ),
            pd.Series(["a", "b"]),
        ),
    ],
)
def test_categorical_input_feature_validate_candidental_invalid(input_feature, values):
    with pytest.raises(ValueError):
        input_feature.validate_candidental(values)


def test_cateogorical_input_is_fulfilled():
    feature = CategoricalInput(
        key="a", categories=["B", "A", "C"], allowed=[True, True, False]
    )
    values = pd.Series(["A", "B", "C", "D"], index=[0, 1, 2, 5])
    fulfilled = feature.is_fulfilled(values)
    assert_series_equal(
        fulfilled,
        pd.Series([True, True, False, False], index=[0, 1, 2, 5]),
    )


@pytest.mark.parametrize(
    "input_feature, expected, expected_value, transform_type",
    [
        (
            CategoricalInput(key="k", categories=categories, allowed=allowed),
            expected,
            expected_value,
            transform_type,
        )
        for categories, allowed, expected, expected_value, transform_type in [
            (["1", "2"], None, False, None, None),
            (["1", "2", "3", "4"], [True, False, False, False], True, ["1"], None),
            (["1", "2", "3", "4"], [True, True, False, True], False, None, None),
            (
                ["1", "2", "3", "4"],
                [True, False, False, False],
                True,
                [0],
                OrdinalEncoding(),
            ),
            (
                ["1", "2", "3", "4"],
                [True, False, False, False],
                True,
                [1, 0, 0, 0],
                OneHotEncoding(),
            ),
            (
                ["1", "2", "3", "4"],
                [True, False, False, False],
                True,
                [0, 0, 0],
                OneHotEncoding(drop_first=True),
            ),
        ]
    ]
    + [
        (
            CategoricalInput(
                key="k",
                categories=["1", "2", "3"],
                allowed=[True, False, False],
                descriptors=Descriptors(
                    columns={"alpha": [1, 3, 5], "beta": [2, 4, 6]}
                ),
            ),
            expected,
            expected_value,
            transform_type,
        )
        for expected, expected_value, transform_type in [
            (True, [1, 2], DescriptorEncoding()),
        ]
    ],
)
def test_categorical_input_feature_is_fixed(
    input_feature,
    expected,
    expected_value,
    transform_type,
):
    assert input_feature.is_fixed() == expected
    if isinstance(transform_type, DescriptorEncoding):
        # the descriptor encoding now carries the descriptor-based fixed value:
        # a fixed (single allowed category) feature has matching lower/upper bounds
        # equal to that category's descriptor row.
        lower, upper = transform_type.get_bounds(input_feature)
        assert lower == expected_value
        assert upper == expected_value
    else:
        assert input_feature.fixed_value(transform_type) == expected_value


@pytest.mark.parametrize(
    "input_feature, expected",
    [
        (
            CategoricalInput(key="k", categories=categories, allowed=allowed),
            expected,
        )
        for categories, allowed, expected in [
            (["a", "b", "c"], [True, True, True], ["a", "b", "c"]),
            (["a", "b", "c"], [False, True, True], ["b", "c"]),
        ]
    ],
)
def test_categorical_input_feature_allowed_categories(input_feature, expected):
    assert input_feature.get_allowed_categories() == expected


@pytest.mark.parametrize(
    "input_feature, expected",
    [
        (
            CategoricalInput(key="k", categories=categories, allowed=allowed),
            expected,
        )
        for categories, allowed, expected in [
            (["a", "b", "c"], [True, True, True], []),
            (["a", "b", "c"], [False, True, True], ["a"]),
        ]
    ],
)
def test_categorical_input_feature_forbidden_categories(input_feature, expected):
    assert input_feature.get_forbidden_categories() == expected


def test_categorical_output_call():
    test_df = pd.DataFrame(data=[[0.7, 0.3], [0.2, 0.8]], columns=["c1", "c2"])
    categorical_output = CategoricalOutput(
        key="a",
        categories=["c1", "c2"],
        objective=ConstrainedCategoricalObjective(
            categories=["c1", "c2"],
            desirability=[True, False],
        ),
    )
    output = categorical_output(test_df, test_df)
    assert output.tolist() == test_df["c1"].tolist()


def test_categorical_input_to_pydantic_field():
    from typing import Literal

    feat = CategoricalInput(key="sol", categories=["water", "ethanol", "toluene"])
    field_type, field_info = feat.to_pydantic_field()
    assert field_type == Literal["water", "ethanol", "toluene"]
    assert (
        field_info.description
        == "Categorical, allowed: ['water', 'ethanol', 'toluene']"
    )


def test_categorical_input_to_pydantic_field_respects_allowed():
    from typing import Literal

    feat = CategoricalInput(
        key="sol",
        categories=["water", "ethanol", "toluene"],
        allowed=[True, True, False],
    )
    field_type, field_info = feat.to_pydantic_field()
    assert field_type == Literal["water", "ethanol"]
    assert field_info.description == "Categorical, allowed: ['water', 'ethanol']"


def test_categorical_input_to_pydantic_field_falls_back_to_str_above_threshold():
    from bofire.data_models.features.categorical import LLM_ENUM_SCHEMA_THRESHOLD

    categories = [f"c{i}" for i in range(LLM_ENUM_SCHEMA_THRESHOLD + 1)]
    feat = CategoricalInput(key="big", categories=categories)
    field_type, field_info = feat.to_pydantic_field()
    assert field_type is str
    # description still lists the categories so the LLM has guidance
    assert "c0" in field_info.description
    assert f"c{LLM_ENUM_SCHEMA_THRESHOLD}" in field_info.description


def test_categorical_input_to_pydantic_field_at_threshold_stays_literal():
    from typing import Literal, get_args, get_origin

    from bofire.data_models.features.categorical import LLM_ENUM_SCHEMA_THRESHOLD

    categories = [f"c{i}" for i in range(LLM_ENUM_SCHEMA_THRESHOLD)]
    feat = CategoricalInput(key="edge", categories=categories)
    field_type, _ = feat.to_pydantic_field()
    assert get_origin(field_type) is get_origin(Literal["x"])
    assert list(get_args(field_type)) == categories


def test_categorical_descriptor_input_to_pydantic_field():
    feat = CategoricalInput(
        key="cat",
        categories=["a", "b"],
        descriptors=Descriptors(columns={"d1": [1.0, 3.0], "d2": [2.0, 4.0]}),
    )
    _, field_info = feat.to_pydantic_field()
    # the values are the point: a model picking a category needs to know what
    # distinguishes them, not merely that a column named d1 exists.
    assert field_info.description == (
        "Categorical, allowed: ['a', 'b'] — "
        "descriptors per category: {'a': {'d1': 1.0, 'd2': 2.0}, "
        "'b': {'d1': 3.0, 'd2': 4.0}}"
    )


def test_categorical_descriptor_input_to_pydantic_field_falls_back_above_threshold():
    from bofire.data_models.features.categorical import LLM_ENUM_SCHEMA_THRESHOLD

    n = LLM_ENUM_SCHEMA_THRESHOLD + 1
    categories = [f"c{i}" for i in range(n)]
    # distinct values per category so the per-descriptor variance validator passes
    feat = CategoricalInput(
        key="big",
        categories=categories,
        descriptors=Descriptors(columns={"d1": [float(i) for i in range(n)]}),
    )
    field_type, field_info = feat.to_pydantic_field()
    assert field_type is str
    # description still lists the categories (via the prefix)
    assert "c0" in field_info.description
    assert f"c{n - 1}" in field_info.description


def test_categorical_with_structure_to_pydantic_field():
    from typing import Literal

    feat = CategoricalInput(
        key="mol",
        categories=["CCO", "CC"],
        descriptors=Descriptors(structure=["CCO", "CC"]),
    )
    field_type, field_info = feat.to_pydantic_field()
    assert field_type == Literal["CCO", "CC"]
    assert field_info.description == (
        "Categorical, allowed: ['CCO', 'CC'] — structure: ['CCO', 'CC']"
    )


def test_categorical_with_structure_to_pydantic_field_structure_beside_names():
    """Categories need not be the SMILES themselves.

    Main could not express this -- `CategoricalMolecularInput` used the categories as the
    structures -- so without the explicit `structure` part the SMILES would be invisible
    to the model here.
    """
    feat = CategoricalInput(
        key="solvent",
        categories=["water", "ethanol"],
        descriptors=Descriptors(columns={"logP": [-1.4, -0.3]}, structure=["O", "CCO"]),
    )
    _, field_info = feat.to_pydantic_field()
    assert field_info.description == (
        "Categorical, allowed: ['water', 'ethanol'] — "
        "descriptors per category: {'water': {'logP': -1.4}, 'ethanol': {'logP': -0.3}} — "
        "structure: ['O', 'CCO']"
    )


def test_categorical_with_structure_to_pydantic_field_falls_back_above_threshold():
    from bofire.data_models.features.categorical import LLM_ENUM_SCHEMA_THRESHOLD

    # Generate enough distinct SMILES by varying alkane chain length
    smiles = ["C" * (i + 1) for i in range(LLM_ENUM_SCHEMA_THRESHOLD + 1)]
    feat = CategoricalInput(
        key="mol", categories=smiles, descriptors=Descriptors(structure=list(smiles))
    )
    field_type, field_info = feat.to_pydantic_field()
    assert field_type is str
    # description still lists the SMILES so the LLM has guidance
    assert smiles[0] in field_info.description
    assert smiles[-1] in field_info.description
