"""Which encoding a surrogate picks when none is given, and where the gate fires.

Lives under `data_models` rather than `tests/bofire/surrogates` on purpose: the
bare-install CI job runs only this tree, and both the default resolution and the gate must
work without the optional `cheminfo` extra.
"""

import pytest

from bofire.data_models.descriptor_generators.api import Fingerprints
from bofire.data_models.encodings.api import DescriptorEncoding
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
