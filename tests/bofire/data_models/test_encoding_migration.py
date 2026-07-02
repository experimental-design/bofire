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
    if expected_type is DescriptorEncoding:
        # "DESCRIPTOR" maps to all static descriptor columns (no generators).
        assert enc.columns is None
        assert enc.generators == {}


def test_migrate_bare_molfeature():
    with pytest.warns(DeprecationWarning):
        migrated = migrate_legacy_encodings({"x": Fingerprints(n_bits=32)})
    enc = migrated["x"]
    assert isinstance(enc, DescriptorEncoding)
    assert enc.columns == []
    assert list(enc.generators) == ["smiles"]
    assert isinstance(enc.generators["smiles"][0], Fingerprints)


def test_migrate_legacy_molecular_encoding_dict():
    """An old serialized ``MolecularEncoding`` dict maps to ``DescriptorEncoding``."""
    from pydantic import TypeAdapter

    from bofire.data_models.encodings.api import AnyCategoricalEncoding

    legacy = {
        "type": "MolecularEncoding",
        "structure": "smiles",
        "generator": Fingerprints(n_bits=32).model_dump(),
    }
    with pytest.warns(DeprecationWarning):
        migrated = migrate_legacy_encodings({"x": legacy})
    # dict-shaped legacy values migrate to a dict; pydantic constructs the object.
    enc = TypeAdapter(AnyCategoricalEncoding).validate_python(migrated["x"])
    assert isinstance(enc, DescriptorEncoding)
    assert enc.columns == []
    assert isinstance(enc.generators["smiles"][0], Fingerprints)


def test_legacy_descriptor_columns_dict_deserializes_natively():
    """The old ``DescriptorEncoding`` dict with ``columns`` is now natively valid."""
    import warnings

    from pydantic import TypeAdapter

    from bofire.data_models.encodings.api import AnyCategoricalEncoding

    legacy = {"type": "DescriptorEncoding", "columns": ["d1", "d2"]}
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # `columns` is a native field, no migration
        migrated = migrate_legacy_encodings({"x": legacy})
    enc = TypeAdapter(AnyCategoricalEncoding).validate_python(migrated["x"])
    assert isinstance(enc, DescriptorEncoding)
    assert enc.columns == ["d1", "d2"]
    assert enc.generators == {}


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
