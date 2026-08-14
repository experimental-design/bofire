"""Tests for the `Descriptors` value object.

How a block is *consumed* -- encoded, decoded, turned into bounds -- lives in
`tests/bofire/data_models/encodings/`. Rejections that a data model raises when
constructed are `add_invalid` specs, not tests here; the SMILES cases below are the
exception, because they need rdkit and specs have no skip mechanism.
"""

import importlib
import re

import pandas as pd
import pytest

from bofire.data_models.features.api import CategoricalInput, ContinuousInput
from bofire.data_models.features.descriptors import Descriptors


RDKIT_AVAILABLE = importlib.util.find_spec("rdkit") is not None

VALID_SMILES = pd.Series(
    ["CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1", "[CH3][CH2][OH]", "N[C@](C)(F)C(=O)O"]
)


# --- the block as a table -------------------------------------------------------


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
    df = f.descriptors.table(list(f.categories))
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
    df = f.descriptors.table([f.key])
    assert len(df.columns) == len(descriptors)
    assert len(df) == 1
    assert df.values.tolist()[0] == values


# --- stacking component blocks --------------------------------------------------


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


# --- structure validation -------------------------------------------------------


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
def test_categorical_with_structure_rejects_invalid_smiles():
    with pytest.raises(ValueError, match="abcd is not a valid smiles string."):
        CategoricalInput(
            key="a",
            categories=["CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1", "abcd"],
            descriptors=Descriptors(
                structure=["CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1", "abcd"]
            ),
        )


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
def test_continuous_with_structure_rejects_invalid_smiles():
    with pytest.raises(ValueError, match="abc is not a valid smiles string"):
        ContinuousInput(
            key="a", bounds=[0, 1], descriptors=Descriptors(structure=["abc"])
        )


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
def test_categorical_with_structure_accepts_valid_smiles():
    feat = CategoricalInput(
        key="a",
        categories=VALID_SMILES.tolist(),
        descriptors=Descriptors(structure=list(VALID_SMILES.tolist())),
    )
    assert feat.descriptors.structure == VALID_SMILES.tolist()
