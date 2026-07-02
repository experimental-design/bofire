"""Migration of legacy encoding spec values to the new encoder objects.

Old serialized surrogates stored ``categorical_encodings`` / ``input_preprocessing_specs``
values as enum strings (``"ONE_HOT"``/``"DUMMY"``/``"ORDINAL"``/``"DESCRIPTOR"``) or as bare
molecular features. These are migrated to the encoder objects on load, with a single
``DeprecationWarning`` per spec map.
"""

import warnings
from typing import Any, Dict

from bofire.data_models.descriptors.generated import GeneratedSource
from bofire.data_models.descriptors.static import StaticSource
from bofire.data_models.encodings.descriptors import DescriptorEncoding
from bofire.data_models.encodings.onehot import OneHotEncoding
from bofire.data_models.encodings.ordinal import OrdinalEncoding
from bofire.data_models.molfeatures.molfeatures import MolFeatures


_MOLFEATURE_TYPES = {
    "Fingerprints",
    "Fragments",
    "MordredDescriptors",
    "CompositeMolFeatures",
}


def _generated_dict(structure: str, generator: dict) -> dict:
    return {
        "type": "DescriptorEncoding",
        "source": {
            "type": "GeneratedSource",
            "structure": structure,
            "generator": generator,
        },
    }


def _migrate_value(value: Any) -> Any:
    """Return the encoder-object form of a legacy value, or the value unchanged.

    A migrated value is always a new object, so callers can detect migration via
    identity (``migrated is not original``).
    """
    legacy = {
        "ONE_HOT": OneHotEncoding,
        "DUMMY": lambda: OneHotEncoding(drop_first=True),
        "ORDINAL": OrdinalEncoding,
        "DESCRIPTOR": lambda: DescriptorEncoding(source=StaticSource()),
    }
    if isinstance(value, str) and value in legacy:
        return legacy[value]()
    if isinstance(value, MolFeatures):
        return DescriptorEncoding(source=GeneratedSource(generator=value))
    if isinstance(value, dict):
        t = value.get("type")
        if t in _MOLFEATURE_TYPES:  # bare molfeature dict
            return _generated_dict("smiles", value)
        if t == "MolecularEncoding":  # pre-source molecular encoding
            return _generated_dict(value.get("structure", "smiles"), value["generator"])
        if t == "DescriptorEncoding" and "columns" in value and "source" not in value:
            return {
                "type": "DescriptorEncoding",
                "source": {"type": "StaticSource", "columns": value["columns"]},
            }
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
            "features / MolecularEncoding) are deprecated; use OneHotEncoding / "
            "OrdinalEncoding / DescriptorEncoding(source=...) instead.",
            DeprecationWarning,
            stacklevel=3,
        )
    return migrated
