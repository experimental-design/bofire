"""Progressive-disclosure navigation of BoFire data model schemas.

``model_json_schema()`` is correct but unusable in a prompt for the larger data
models: ``SoboStrategy`` produces ~45,500 tokens across 120 ``$defs``. This
package lets a caller walk the schema instead of dumping it -- ten direct fields
first, then the members that fit a chosen field, then that member's own fields.

    >>> from bofire.data_models.introspection.api import describe
    >>> describe("SoboStrategy")["fields"]["acquisition_function"]["options"][:3]
    ['qNEI', 'qEI', 'qSR']
    >>> describe("qUCB")["fields"]["beta"]["type"]
    'float'

Paths address locations inside a model and are shared between reading and, in
future, writing, so a path reported by ``describe`` can be handed straight back
to a mutating call. See :mod:`bofire.data_models.introspection.paths` for the
grammar.
"""

from typing import Any, Dict, Optional

from bofire.data_models.introspection.paths import (
    PathError,
    PathSegment,
    SegmentKind,
    parse_path,
    unparse,
)
from bofire.data_models.introspection.registry import (
    AmbiguousNameError,
    UnknownNameError,
    ambiguous_names,
    lookup,
)
from bofire.data_models.introspection.render import (
    approx_tokens,
    render,
    shallow_schema,
)
from bofire.data_models.introspection.resolver import Cursor, resolve


__all__ = [
    "AmbiguousNameError",
    "Cursor",
    "PathError",
    "PathSegment",
    "SegmentKind",
    "UnknownNameError",
    "ambiguous_names",
    "approx_tokens",
    "describe",
    "full_schema",
    "lookup",
    "parse_path",
    "resolve",
    "unparse",
]


def describe(
    path: str,
    instance: Any = None,
    root: Optional[str] = None,
    full: bool = False,
) -> Dict[str, Any]:
    """Describe the location *path* addresses.

    Args:
        path: What to describe. Without *instance* the first segment names a
            type, as in ``"SoboStrategy.acquisition_function"``. With *instance*
            the whole path is relative to it, as in
            ``"domain.inputs.temperature.bounds"``; ``""`` is the instance itself.
        instance: A data model object or a serialized spec dict. Supplying one
            resolves element keys, collapses unions to the member actually
            present, and reports current values.
        root: The type name of *instance*, needed only when it is a dict that
            carries no ``type`` discriminator.
        full: Return the node's complete JSON schema instead of a summary. The
            schema is shallow: ``$defs`` are dropped and each ``$ref`` is left as
            a bare type name that ``describe`` accepts, so the cost stays bounded
            by the node's own fields rather than by everything reachable from it.

    Returns:
        A dict whose ``kind`` is one of ``class``, ``union``, ``keyed_container``,
        ``list``, ``mapping``, ``leaf``, ``enum`` or ``literal``. Keys with
        nothing to report are omitted rather than set to ``None``.

    Raises:
        PathError: If the path is malformed or does not resolve.
        UnknownNameError: If a named type is not registered.
        AmbiguousNameError: If a named type is defined in more than one place.
    """
    return render(resolve(path, instance=instance, root=root), full=full)


def full_schema(name: str) -> Dict[str, Any]:
    """The shallow JSON schema of a named type.

    A shortcut for ``describe(name, full=True)`` when the caller already knows
    the type name and has no path to walk.
    """
    return shallow_schema(lookup(name))
