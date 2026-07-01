"""Migration of legacy encoding spec values to the new encoder objects.

Old serialized surrogates stored ``categorical_encodings`` / ``input_preprocessing_specs``
values as enum strings (``"ONE_HOT"``/``"DUMMY"``/``"ORDINAL"``/``"DESCRIPTOR"``) or as bare
molecular features. These are migrated to the encoder objects on load, with a
``DeprecationWarning``. Import-light (no ``features`` import) to avoid import cycles.
"""

import warnings
from typing import Any, Dict


_MOLFEATURE_TYPES = {
    "Fingerprints",
    "Fragments",
    "MordredDescriptors",
    "CompositeMolFeatures",
}


def _warn(what: str) -> None:
    warnings.warn(
        f"Legacy categorical encoding value {what} is deprecated; use the encoding "
        "objects (OneHotEncoding / OrdinalEncoding / DescriptorEncoding / "
        "MolecularEncoding) instead.",
        DeprecationWarning,
        stacklevel=3,
    )


def _migrate_value(value: Any) -> Any:
    from bofire.data_models.encodings.descriptors import DescriptorEncoding
    from bofire.data_models.encodings.molecular import MolecularEncoding
    from bofire.data_models.encodings.onehot import OneHotEncoding
    from bofire.data_models.encodings.ordinal import OrdinalEncoding
    from bofire.data_models.molfeatures.molfeatures import MolFeatures

    legacy = {
        "ONE_HOT": OneHotEncoding,
        "DUMMY": lambda: OneHotEncoding(drop_first=True),
        "ORDINAL": OrdinalEncoding,
        "DESCRIPTOR": DescriptorEncoding,
    }
    if isinstance(value, str) and value in legacy:
        _warn(repr(value))
        return legacy[value]()
    if isinstance(value, MolFeatures):
        _warn("a bare molecular feature")
        return MolecularEncoding(generator=value)
    if isinstance(value, dict) and value.get("type") in _MOLFEATURE_TYPES:
        _warn("a bare molecular feature")
        return {"type": "MolecularEncoding", "generator": value}
    return value


def migrate_legacy_encodings(specs: Dict[str, Any]) -> Dict[str, Any]:
    """Return ``specs`` with any legacy values migrated to encoder objects."""
    if not isinstance(specs, dict):
        return specs
    return {key: _migrate_value(value) for key, value in specs.items()}
