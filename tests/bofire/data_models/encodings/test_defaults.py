"""Which encoding a surrogate picks when no explicit one is given.

Lives under `data_models` rather than `tests/bofire/surrogates` on purpose: the
bare-install CI job runs only this tree, and resolving a default must work without the
optional `cheminfo` extra.
"""

import pytest

from bofire.data_models.features.api import CategoricalInput
from bofire.data_models.features.descriptors import Descriptors


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
