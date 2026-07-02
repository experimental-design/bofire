"""Migration of legacy encoding spec values to the new encoder objects.

Old serialized surrogates stored ``categorical_encodings`` / ``input_preprocessing_specs``
values as enum strings (``"ONE_HOT"``/``"DUMMY"``/``"ORDINAL"``/``"DESCRIPTOR"``), as bare
molecular features, or as a ``MolecularEncoding``. These are migrated to
``DescriptorEncoding`` (with its flat ``columns`` / ``generators`` fields) on load, with a
single ``DeprecationWarning`` per spec map.
"""

import warnings
from typing import Any, Dict

from bofire.data_models.encodings.descriptors import DescriptorEncoding
from bofire.data_models.encodings.onehot import OneHotEncoding
from bofire.data_models.encodings.ordinal import OrdinalEncoding
from bofire.data_models.molfeatures.molfeatures import MolFeatures


_MOLFEATURE_TYPES = {"Fingerprints", "Fragments", "MordredDescriptors"}


def _generator_list(generator: Any) -> list:
    """Expand a (possibly legacy composite) molfeature into a flat generator list."""
    if isinstance(generator, dict) and generator.get("type") == "CompositeMolFeatures":
        return generator["features"]
    return [generator]


def _descriptor_dict(structure: str, generator: Any) -> dict:
    return {
        "type": "DescriptorEncoding",
        "columns": [],
        "generators": {structure: _generator_list(generator)},
    }


def _migrate_value(value: Any) -> Any:
    """Return the encoder-object form of a legacy value, or the value unchanged.

    A migrated value is always a new object/dict, so callers can detect migration via
    identity (``migrated is not original``).
    """
    legacy = {
        "ONE_HOT": OneHotEncoding,
        "DUMMY": lambda: OneHotEncoding(drop_first=True),
        "ORDINAL": OrdinalEncoding,
        "DESCRIPTOR": DescriptorEncoding,  # all numeric descriptor columns
    }
    if isinstance(value, str) and value in legacy:
        return legacy[value]()
    if isinstance(value, MolFeatures):  # bare generator
        return DescriptorEncoding(columns=[], generators={"smiles": [value]})
    if isinstance(value, dict):
        t = value.get("type")
        if (
            t in _MOLFEATURE_TYPES or t == "CompositeMolFeatures"
        ):  # bare molfeature dict
            return _descriptor_dict("smiles", value)
        if t == "MolecularEncoding":  # pre-source molecular encoding
            return _descriptor_dict(
                value.get("structure", "smiles"), value["generator"]
            )
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
            "OrdinalEncoding / DescriptorEncoding(columns=..., generators=...) instead.",
            DeprecationWarning,
            stacklevel=3,
        )
    return migrated
