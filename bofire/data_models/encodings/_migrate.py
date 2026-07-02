"""Migration of legacy encoding spec values to the new encoder objects.

Old serialized surrogates stored ``categorical_encodings`` / ``input_preprocessing_specs``
values as enum strings (``"ONE_HOT"``/``"DUMMY"``/``"ORDINAL"``/``"DESCRIPTOR"``) or as bare
molecular features. These are migrated to the encoder objects on load, with a single
``DeprecationWarning`` per spec map. Import-light (no ``features`` import) to avoid cycles.
"""

import warnings
from typing import Any, Dict


_MOLFEATURE_TYPES = {
    "Fingerprints",
    "Fragments",
    "MordredDescriptors",
    "CompositeMolFeatures",
}


def _migrate_value(value: Any) -> Any:
    """Return the encoder-object form of a legacy value, or the value unchanged.

    A migrated value is always a new object, so callers can detect migration via
    identity (``migrated is not original``).
    """
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
        return legacy[value]()
    if isinstance(value, MolFeatures):
        return MolecularEncoding(generator=value)
    if isinstance(value, dict) and value.get("type") in _MOLFEATURE_TYPES:
        return {"type": "MolecularEncoding", "generator": value}
    return value


def migrate_legacy_encodings(specs: Dict[str, Any]) -> Dict[str, Any]:
    """Return ``specs`` with any legacy values migrated to encoder objects.

    Emits a single ``DeprecationWarning`` if at least one value was migrated.
    """
    if not isinstance(specs, dict):
        return specs
    migrated = {}
    changed = False
    for key, value in specs.items():
        new_value = _migrate_value(value)
        changed = changed or new_value is not value
        migrated[key] = new_value
    if changed:
        warnings.warn(
            "Legacy categorical encoding values (enum strings / bare molecular "
            "features) are deprecated; use the encoding objects (OneHotEncoding / "
            "OrdinalEncoding / DescriptorEncoding / MolecularEncoding) instead.",
            DeprecationWarning,
            stacklevel=3,
        )
    return migrated
