"""Progressive-disclosure navigation of BoFire data model schemas.

``model_json_schema()`` is correct but too large to put in a prompt for the
bigger data models (``SoboStrategy``: ~45,500 tokens). ``describe`` walks the
schema one node at a time instead:

    >>> from bofire.data_models.introspection.api import describe
    >>> describe(["SoboStrategy"])["fields"]["acquisition_function"]["options"][:3]
    ['qNEI', 'qEI', 'qSR']
    >>> describe(["qUCB"])["fields"]["beta"]["type"]
    'float'
"""

from typing import Any, Dict, Optional, Sequence, Union

from bofire.data_models.introspection.registry import (
    AmbiguousNameError,
    UnknownNameError,
    lookup,
)
from bofire.data_models.introspection.render import (
    approx_tokens,
    render,
    shallow_schema,
)
from bofire.data_models.introspection.resolver import PathError, resolve


__all__ = [
    "AmbiguousNameError",
    "PathError",
    "UnknownNameError",
    "approx_tokens",
    "describe",
    "full_schema",
    "lookup",
]


def describe(
    path: Sequence[Union[str, int]],
    spec: Optional[Dict[str, Any]] = None,
    root: Optional[str] = None,
    full: bool = False,
) -> Dict[str, Any]:
    """Describe the location *path* addresses.

    Args:
        path: A list of segments. Strings name a field, a union member or an
            element key; integers name a list position. Without *spec* the first
            segment is a type name, e.g. ``["SoboStrategy",
            "acquisition_function"]``. With *spec* the path is relative to it,
            e.g. ``["domain", "inputs", "temperature", "bounds"]``, and ``[]`` is
            the spec itself.
        spec: A serialized data model, as produced by ``model_dump(mode="json")``.
            Supplies element keys, the union member actually present, and
            current values.
        root: The type name of *spec*, needed only when it has no ``type`` key.
        full: Return the node's JSON schema instead of a summary. ``$defs`` are
            dropped and each ``$ref`` is left as a type name ``describe`` accepts.

    Returns:
        A dict whose ``kind`` is ``class``, ``union``, ``list`` or ``leaf``. Keys
        with nothing to report are omitted.

    Raises:
        PathError: If the path is malformed or does not resolve.
        UnknownNameError: If a type name is not registered.
        AmbiguousNameError: If a type name is defined in more than one place.
    """
    return render(resolve(path, spec=spec, root=root), full=full)


def full_schema(name: str) -> Dict[str, Any]:
    """The shallow JSON schema of a named type; see ``describe(..., full=True)``."""
    return shallow_schema(lookup(name))
