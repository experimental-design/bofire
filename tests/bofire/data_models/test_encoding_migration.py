"""Tests for migration of legacy categorical-encoding spec values.

`CategoricalEncodingEnum` was removed in favour of encoder objects. Legacy
serialized surrogates stored string values (``"ONE_HOT"`` etc.) or bare molecular
features; these must still deserialize, emit a ``DeprecationWarning``, and re-emit
as encoder objects.
"""

import pytest

from bofire.data_models.encodings._migrate import migrate_legacy_encodings
from bofire.data_models.encodings.api import (
    DescriptorEncoding,
    MolecularEncoding,
    OneHotEncoding,
    OrdinalEncoding,
)
from bofire.data_models.molfeatures.api import Fingerprints


@pytest.mark.parametrize(
    "legacy, expected_type, drop_first",
    [
        ("ONE_HOT", OneHotEncoding, False),
        ("DUMMY", OneHotEncoding, True),
        ("ORDINAL", OrdinalEncoding, False),
        ("DESCRIPTOR", DescriptorEncoding, False),
    ],
)
def test_migrate_legacy_string_values(legacy, expected_type, drop_first):
    with pytest.warns(DeprecationWarning):
        migrated = migrate_legacy_encodings({"x": legacy})
    enc = migrated["x"]
    assert isinstance(enc, expected_type)
    if expected_type is OneHotEncoding:
        assert enc.drop_first is drop_first


def test_migrate_bare_molfeature():
    with pytest.warns(DeprecationWarning):
        migrated = migrate_legacy_encodings({"x": Fingerprints(n_bits=32)})
    assert isinstance(migrated["x"], MolecularEncoding)
    assert isinstance(migrated["x"].generator, Fingerprints)


def test_migrate_passthrough_objects_no_warning():
    import warnings

    specs = {"a": OneHotEncoding(), "b": OrdinalEncoding()}
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no DeprecationWarning expected
        migrated = migrate_legacy_encodings(specs)
    assert migrated == specs


def test_surrogate_deserializes_legacy_string_specs():
    """An old serialized surrogate with string encodings loads and re-emits objects."""
    from bofire.data_models.domain.api import Inputs, Outputs
    from bofire.data_models.features.api import (
        CategoricalInput,
        ContinuousInput,
        ContinuousOutput,
    )
    from bofire.data_models.surrogates.api import SingleTaskGPSurrogate

    inputs = Inputs(
        features=[
            CategoricalInput(key="cat", categories=["a", "b", "c"]),
            ContinuousInput(key="x", bounds=(0, 1)),
        ]
    )
    outputs = Outputs(features=[ContinuousOutput(key="y")])

    with pytest.warns(DeprecationWarning):
        surrogate = SingleTaskGPSurrogate(
            inputs=inputs,
            outputs=outputs,
            categorical_encodings={"cat": "ONE_HOT"},
        )
    assert isinstance(surrogate.categorical_encodings["cat"], OneHotEncoding)
    # input_preprocessing_specs is forced to ordinal for botorch surrogates.
    assert isinstance(surrogate.input_preprocessing_specs["cat"], OrdinalEncoding)
    # re-emits in the new object shape
    dump = surrogate.model_dump()
    assert dump["categorical_encodings"]["cat"] == {
        "type": "OneHotEncoding",
        "drop_first": False,
    }
