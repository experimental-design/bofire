import random
import re

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import tests.bofire.data_models.specs.api as specs
from bofire.data_models.descriptor_generators.api import Fingerprints
from bofire.data_models.encodings.api import DescriptorEncoding
from bofire.data_models.features.api import CategoricalInput, ContinuousInput
from bofire.data_models.features.descriptors import Descriptors


@pytest.mark.parametrize(
    "key, categories, samples_in, descriptors",
    [
        ("c", ["B", "A", "C"], ["A", "A", "C", "B"], ["d1", "d2"]),
        (
            "c_alpha",
            ["B_b", "_A_a", "C_c_"],
            ["_A_a", "_A_a", "C_c_", "B_b"],
            ["_d1_d", "d2_d_2_"],
        ),
        (
            "__c_alpha_c_",
            ["__c_alpha_c__B_b", "__c_alpha_c___A_a", "__c_alpha_c__C_c_"],
            [
                "__c_alpha_c___A_a",
                "__c_alpha_c___A_a",
                "__c_alpha_c__C_c_",
                "__c_alpha_c__B_b",
            ],
            ["__c_alpha_c__d1_d", "__c_alpha_c_d2_d_2_"],
        ),
    ],
)
def test_categorical_descriptor_to_descriptor_encoding(
    key,
    categories,
    samples_in,
    descriptors,
):
    c = CategoricalInput(
        key=key,
        categories=categories,
        descriptors=Descriptors(
            columns={descriptors[0]: [1, 3, 5], descriptors[1]: [2, 4, 6]}
        ),
    )
    samples = pd.Series(samples_in)
    t_samples = DescriptorEncoding().encode(c, samples)
    assert_frame_equal(
        t_samples,
        pd.DataFrame(
            data=[[3.0, 4.0], [3.0, 4.0], [5.0, 6.0], [1.0, 2.0]],
            columns=[f"{key}_{des_str}" for des_str in descriptors],
        ),
    )
    untransformed = DescriptorEncoding().decode(c, t_samples)
    assert np.all(samples == untransformed)


@pytest.mark.parametrize(
    "key, categories, descriptors",
    [
        ("c", ["B", "A", "C"], ["d1", "d2"]),
        ("c_alpha", ["B_b", "_A_a", "C_c_"], ["_d1_d", "d2_d_2_"]),
        (
            "__c_alpha_c_",
            ["__c_alpha_c__B_b", "__c_alpha_c___A_a", "__c_alpha_c__C_c_"],
            ["__c_alpha_c__d1_d", "__c_alpha_c_d2_d_2_"],
        ),
    ],
)
def test_categorical_descriptor_from_descriptor_encoding(key, categories, descriptors):
    c1 = CategoricalInput(
        key=key,
        categories=categories,
        descriptors=Descriptors(
            columns={descriptors[0]: [1, 3, 5], descriptors[1]: [2, 4, 6]}
        ),
    )
    descriptor_values = pd.DataFrame(
        columns=[f"{key}_{des_str}" for des_str in descriptors] + ["misc"],
        data=[[1.05, 2.5, 6], [4, 4.5, 9]],
    )
    samples = DescriptorEncoding().decode(c1, descriptor_values)
    print(samples)
    assert np.all(samples == pd.Series([categories[0], categories[1]]))

    c2 = CategoricalInput(
        key=key,
        categories=categories,
        descriptors=Descriptors(
            columns={descriptors[0]: [1, 3, 5], descriptors[1]: [2, 4, 6]}
        ),
        allowed=[False, True, True],
    )

    samples = DescriptorEncoding().decode(c2, descriptor_values)
    print(samples)
    assert np.all(samples == pd.Series([categories[1], categories[1]]))


@pytest.mark.parametrize(
    "input_feature, expected_with_values, expected",
    [
        (
            CategoricalInput(
                key="if1",
                categories=["a", "b"],
                allowed=[True, True],
                descriptors=Descriptors(columns={"alpha": [1, 3], "beta": [2, 4]}),
            ),
            ([1, 2], [3, 4]),
            ([1, 2], [3, 4]),
        ),
        (
            CategoricalInput(
                key="if2",
                categories=["a", "b", "c"],
                allowed=[True, False, True],
                descriptors=Descriptors(
                    columns={"alpha": [1, 3, 1], "beta": [2, 4, 5]}
                ),
            ),
            ([1, 2], [3, 5]),
            ([1, 2], [1, 5]),
        ),
        # (CategoricalInput(key="if2", categories = ["a","b"], allowed = [True, True]), ["a","b"]),
        # (CategoricalInput(key="if3", categories = ["a","b"], allowed = [True, False]), ["a"]),
        # (CategoricalInput(key="if4", categories = ["a","b"], allowed = [True, False]), ["a", "b"]),
        # (ContinuousInput(key="if1", lower_bound=2.5, upper_bound=2.9), (1,3.)),
        # (ContinuousInput(key="if2", lower_bound=1., upper_bound=3.), (1,3.)),
        # (ContinuousInput(key="if2", lower_bound=1., upper_bound=1.), (1,1.)),
    ],
)
def test_categorical_descriptor_feature_get_bounds(
    input_feature,
    expected_with_values,
    expected,
):
    experiments = pd.DataFrame(
        {"if1": ["a", "b"], "if2": ["a", "c"], "if3": ["a", "a"], "if4": ["b", "b"]},
    )
    lower, upper = DescriptorEncoding().get_bounds(
        input_feature,
        values=experiments[input_feature.key],
    )
    assert np.allclose(lower, expected_with_values[0])
    assert np.allclose(upper, expected_with_values[1])
    lower, upper = DescriptorEncoding().get_bounds(
        input_feature,
        values=None,
    )
    assert np.allclose(lower, expected[0])
    assert np.allclose(upper, expected[1])


@pytest.mark.parametrize(
    "input_feature, values, strict",
    [
        (
            specs.features.valid(CategoricalInput).obj(
                categories=["c1", "c2", "c3"],
            ),
            pd.Series([random.choice(["c1", "c2", "c3"]) for _ in range(20)]),
            True,
        ),
        (
            specs.features.valid(CategoricalInput).obj(
                categories=["c1", "c2", "c3"],
            ),
            pd.Series([random.choice(["c1", "c2", "c3"]) for _ in range(20)]),
            False,
        ),
        (
            specs.features.valid(CategoricalInput).obj(
                categories=["c1", "c2", "c3"],
                allowed=[True, False, False],
            ),
            pd.Series([random.choice(["c1", "c2", "c3"]) for _ in range(200)]),
            True,
        ),
        (
            specs.features.valid(CategoricalInput).obj(
                categories=["c1", "c2", "c3"],
                allowed=[True, False, False],
            ),
            pd.Series([random.choice(["c1", "c2", "c3"]) for _ in range(200)]),
            False,
        ),
        (
            specs.features.valid(CategoricalInput).obj(
                categories=["c1", "c2", "c3"],
                allowed=[True, False, False],
            ),
            pd.Series(["c1", "c1"]),
            False,
        ),
        (
            specs.features.valid(CategoricalInput).obj(
                categories=["c1", "c2", "c3"],
                allowed=[False, True, True],
                descriptors=Descriptors(columns={"d1": [1, 3, 3], "d2": [2, 7, 1]}),
            ),
            pd.Series(["c2", "c3"]),
            False,
        ),
        (
            specs.features.valid(CategoricalInput).obj(
                categories=["1", "2", "3"],
                allowed=[True, False, False],
            ),
            pd.Series([random.choice([1, 2, 3]) for _ in range(200)]),
            False,
        ),
    ],
)
def test_categorical_descriptor_input_feature_validate_valid(
    input_feature,
    values,
    strict,
):
    input_feature.validate_experimental(values, strict)


@pytest.mark.parametrize(
    "input_feature, values, strict",
    [
        (
            specs.features.valid(CategoricalInput).obj(),
            pd.Series(["c1", "c4"]),
            True,
        ),
        (
            specs.features.valid(CategoricalInput).obj(),
            pd.Series(["c1", "c4"]),
            False,
        ),
        (
            specs.features.valid(CategoricalInput).obj(
                categories=["c1", "c2", "c3"],
                allowed=[True, False, False],
                descriptors=Descriptors(columns={"d1": [1, 3, 5], "d2": [2, 7, 1]}),
            ),
            pd.Series(["c1", "c1"]),
            True,
        ),
        (
            specs.features.valid(CategoricalInput).obj(
                categories=["c1", "c2", "c3"],
                allowed=[False, True, True],
                descriptors=Descriptors(columns={"d1": [1, 3, 3], "d2": [2, 7, 1]}),
            ),
            pd.Series(["c2", "c3"]),
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
    ],
)
def test_categorical_descriptor_input_feature_validate_invalid(
    input_feature,
    values,
    strict,
):
    with pytest.raises(ValueError):
        input_feature.validate_experimental(values, strict)


@pytest.mark.parametrize(
    "categories, descriptors, values",
    [
        (["c1", "c2"], ["d1", "d2", "d3"], [[1, 2, 3], [4, 5, 6]]),
        (
            ["c1", "c2", "c3", "c4"],
            ["d1", "d2", "d3"],
            [
                [1, 2, 3],
                [4, 5, 6],
                [4, 5, 6],
                [4, 5, 6],
            ],
        ),
    ],
)
def test_categorical_descriptor_input_feature_as_dataframe(
    categories,
    descriptors,
    values,
):
    f = CategoricalInput(
        key="k",
        categories=categories,
        descriptors=Descriptors(
            columns={
                name: [row[j] for row in values] for j, name in enumerate(descriptors)
            }
        ),
    )
    df = f.descriptors.table(f.descriptor_levels())
    assert len(df.columns) == len(descriptors)
    assert len(df) == len(categories)
    assert df.values.tolist() == values


@pytest.mark.parametrize(
    "descriptors, values",
    [
        (["a", "b"], [1.0, 2.0]),
        (["a", "b", "c"], [1.0, 2.0, 3.0]),
    ],
)
def test_continuous_descriptor_input_feature_as_dataframe(descriptors, values):
    f = ContinuousInput(
        key="k",
        bounds=(1, 2),
        descriptors=Descriptors(
            columns={name: [values[j]] for j, name in enumerate(descriptors)}
        ),
    )
    df = f.descriptors.table(f.descriptor_levels())
    assert len(df.columns) == len(descriptors)
    assert len(df) == 1
    assert df.values.tolist()[0] == values


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


def test_continuous_descriptor_input_to_pydantic_field():
    feat = ContinuousInput(
        key="x", bounds=(0, 1), descriptors=Descriptors(columns={"d1": [0.5]})
    )
    field_type, field_info = feat.to_pydantic_field()
    assert field_type is float
    # one value per descriptor: the feature is a single component, not a table
    assert (
        field_info.description
        == "Continuous, bounds [0.0, 1.0] — descriptors: {'d1': 0.5}"
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


@pytest.mark.parametrize(
    "descriptors, expected_columns, expected_generators",
    [
        # a structure alone auto-enables fingerprints
        (Descriptors(structure=["CCO", "CC"]), None, ["Fingerprints"]),
        # numeric columns alone use a static encoding
        (Descriptors(columns={"logP": [-0.3, 1.8]}), None, []),
        # both: the handcrafted columns must survive alongside the fingerprints.
        # This used to resolve to columns=[], silently dropping them.
        (
            Descriptors(columns={"logP": [-0.3, 1.8]}, structure=["CCO", "CC"]),
            None,
            ["Fingerprints"],
        ),
    ],
    ids=["structure-only", "columns-only", "both"],
)
def test_default_encoding_keeps_all_descriptor_data(
    descriptors, expected_columns, expected_generators
):
    from bofire.data_models.surrogates.api import SingleTaskGPSurrogate

    feat = CategoricalInput(key="c", categories=["CCO", "CC"], descriptors=descriptors)
    encoding = SingleTaskGPSurrogate._resolve_default_categorical_encoding(feat)
    # columns=None means "every numeric column the feature carries"
    assert encoding.columns == expected_columns
    assert [type(g).__name__ for g in encoding.generators] == expected_generators


def test_descriptors_concat_stacks_component_blocks():
    """The one-row blocks of mixture components stack into one block."""
    a = Descriptors(columns={"logP": [-0.3], "MW": [46.0]}, structure=["CCO"])
    # declared in a different order — the merged block follows the *first* block, because
    # filter_correlated keeps the first of each correlated group
    b = Descriptors(columns={"MW": [18.0], "logP": [-1.4]}, structure=["O"])
    merged = Descriptors.concat([a, b])
    assert merged.names == ["logP", "MW"]
    assert merged.columns == {"logP": [-0.3, -1.4], "MW": [46.0, 18.0]}
    assert merged.structure == ["CCO", "O"]
    assert len(merged) == 2


@pytest.mark.parametrize(
    "blocks, message",
    [
        (
            [
                Descriptors(columns={"logP": [1.0], "MW": [2.0]}),
                Descriptors(columns={"logP": [3.0]}),
            ],
            "same descriptor columns",
        ),
        (
            [
                Descriptors(columns={"logP": [1.0]}, structure=["CCO"]),
                Descriptors(columns={"logP": [3.0]}),
            ],
            "either all components carry a `structure` or none do",
        ),
        ([Descriptors(columns={"logP": [1.0]}), None], "carry no descriptors"),
        ([], "empty list"),
    ],
    ids=["mismatched-columns", "mixed-structure", "none-entry", "empty"],
)
def test_descriptors_concat_rejects_inconsistent_blocks(blocks, message):
    """Previously a mismatch surfaced as a bare ``KeyError`` from deep inside pandas."""
    with pytest.raises(ValueError, match=re.escape(message)):
        Descriptors.concat(blocks)


@pytest.mark.parametrize(
    "spec, descriptors",
    [
        # two generators emitting the same names
        (
            DescriptorEncoding(
                columns=[],
                generators=[Fingerprints(n_bits=8), Fingerprints(n_bits=8)],
            ),
            Descriptors(structure=["CCO", "CC"]),
        ),
        # a static column colliding with a generated one
        (
            DescriptorEncoding(generators=[Fingerprints(n_bits=8)]),
            Descriptors(columns={"fingerprint_0": [1.0, 2.0]}, structure=["CCO", "CC"]),
        ),
    ],
    ids=["two-identical-generators", "static-collides-with-generated"],
)
def test_duplicate_descriptor_names_are_caught_by_the_gate(spec, descriptors):
    """Duplicates used to escape validation and only surface later, at transform time."""
    with pytest.raises(ValueError, match="descriptor names must be unique"):
        spec.validate_for(descriptors, "c")


def test_duplicate_descriptor_names_fail_at_surrogate_construction():
    """Because the gate runs from BotorchSurrogate, the failure is early and located."""
    from bofire.data_models.domain.api import Inputs, Outputs
    from bofire.data_models.features.api import ContinuousOutput
    from bofire.data_models.surrogates.api import SingleTaskGPSurrogate

    feature = CategoricalInput(
        key="c",
        categories=["CCO", "CC"],
        descriptors=Descriptors(structure=["CCO", "CC"]),
    )
    with pytest.raises(ValueError, match="descriptor names must be unique"):
        SingleTaskGPSurrogate(
            inputs=Inputs(features=[feature]),
            outputs=Outputs(features=[ContinuousOutput(key="y")]),
            categorical_encodings={
                "c": DescriptorEncoding(
                    columns=[],
                    generators=[Fingerprints(n_bits=8), Fingerprints(n_bits=8)],
                )
            },
        )
