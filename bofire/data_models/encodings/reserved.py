"""Registry of reserved descriptor column keys for categorical features.

Most descriptor columns on a feature are free-form numeric properties. A small
set of *reserved* keys carry known semantics and are validated + role-shielded:
e.g. ``smiles`` is a structure identifier (consumed by molecular encoders) rather
than a numeric descriptor, so it must not be fed to a numeric encoder.

The registry is intentionally lightweight (a name -> record dict) and extensible
via :func:`register_reserved_descriptor`. It is kept import-light (no feature or
encoding imports, cheminformatics imported lazily) so ``data_models`` stays
importable without rdkit and there is no ``features <-> encodings`` import cycle.
"""

import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Literal


DescriptorRole = Literal["descriptor", "structure"]


@dataclass(frozen=True)
class ReservedDescriptor:
    """A descriptor column name with known semantics.

    Attributes:
        name: The reserved column name.
        dtype: The expected python type of each entry (e.g. ``str`` or ``float``).
        role: ``"structure"`` for identifiers consumed by generators (e.g. SMILES),
            ``"descriptor"`` for directly usable numeric columns.
        validator: Callable validating the column values; raises on invalid input.
    """

    name: str
    dtype: type
    role: DescriptorRole
    validator: Callable[[List], None]


_RESERVED: Dict[str, ReservedDescriptor] = {}


def register_reserved_descriptor(reserved: ReservedDescriptor) -> None:
    """Register a reserved descriptor key.

    Raises:
        ValueError: If a *different* reserved descriptor with the same name is
            already registered.
    """
    existing = _RESERVED.get(reserved.name)
    if existing is not None and existing != reserved:
        raise ValueError(
            f"A different reserved descriptor with name '{reserved.name}' "
            "is already registered.",
        )
    _RESERVED[reserved.name] = reserved


def is_reserved(name: str) -> bool:
    """Whether ``name`` is a reserved descriptor key."""
    return name in _RESERVED


def get_reserved_descriptor(name: str) -> ReservedDescriptor:
    """Return the :class:`ReservedDescriptor` record for ``name``."""
    return _RESERVED[name]


def _validate_smiles_column(values: List) -> None:
    """Validate that each entry is a parseable SMILES.

    No-op (with a warning) when rdkit is not available, mirroring the behaviour
    of ``CategoricalMolecularInput.validate_smiles``.
    """
    from bofire.utils.cheminformatics import smiles2mol

    try:
        smiles2mol(values[0])
    except NameError:
        warnings.warn("rdkit not installed, smiles descriptors cannot be validated.")
        return
    for value in values:
        smiles2mol(value)


register_reserved_descriptor(
    ReservedDescriptor(
        name="smiles",
        dtype=str,
        role="structure",
        validator=_validate_smiles_column,
    ),
)
