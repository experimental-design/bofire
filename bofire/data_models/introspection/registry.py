"""Lazy index mapping a name to the data model class it denotes.

``describe("qUCB")`` has to turn a bare string into a class. The index is built
by walking ``bofire.data_models`` with ``pkgutil``, the same enumeration the
documentation ratchet uses, so the two stay in lockstep. It is deliberately not
built from ``bofire.data_models.api.AnyThing``: that list is assembled from a
handful of unions and omits reachable types such as ``BotorchSurrogates``,
``Step``, the encodings, the transforms and the convergence criteria, so
navigation would dead-end on them.

The walk imports every module under ``bofire.data_models`` and is therefore
neither cheap nor safe to run at import time -- it would re-enter this package.
It runs once, on first use, behind :func:`functools.lru_cache`.
"""

import importlib
import pkgutil
from collections import defaultdict
from functools import lru_cache
from typing import Dict, FrozenSet, Tuple

from pydantic import BaseModel

from bofire.data_models.introspection.annotations import type_tag


__all__ = ["AmbiguousNameError", "UnknownNameError", "ambiguous_names", "lookup"]

_SELF_PACKAGE = "bofire.data_models.introspection"


class UnknownNameError(KeyError):
    """Raised when a name does not denote any data model."""


class AmbiguousNameError(KeyError):
    """Raised when a name denotes more than one data model."""


def _is_indexable(obj: object) -> bool:
    """True if *obj* is a data model class worth indexing under its own name."""
    import enum

    if not isinstance(obj, type):
        return False
    if not (issubclass(obj, BaseModel) or issubclass(obj, enum.Enum)):
        return False
    if not getattr(obj, "__module__", "").startswith("bofire.data_models"):
        return False
    # Private helpers such as ``_BaseFeatures`` are implementation detail.
    if obj.__name__.startswith("_"):
        return False
    # Parametrized pydantic generics (``_BaseFeatures[Annotated[Union[...]]]``)
    # surface with a type expression for a name and duplicate their origin's tag.
    metadata = getattr(obj, "__pydantic_generic_metadata__", None)
    if metadata and metadata.get("args"):
        return False
    return True


@lru_cache(maxsize=1)
def _index() -> Dict[str, Tuple[type, ...]]:
    """Build the name index. Cached; the walk runs at most once per process."""
    import bofire.data_models

    found: Dict[str, set] = defaultdict(set)
    for module_info in pkgutil.walk_packages(
        bofire.data_models.__path__, bofire.data_models.__name__ + "."
    ):
        if module_info.name.startswith(_SELF_PACKAGE):
            continue
        module = importlib.import_module(module_info.name)
        for obj in vars(module).values():
            if not _is_indexable(obj):
                continue
            found[obj.__name__].add(obj)
            tag = type_tag(obj)
            if tag:
                found[tag].add(obj)
            found[f"{obj.__module__}.{obj.__qualname__}"].add(obj)
    return {
        name: tuple(sorted(v, key=lambda c: c.__name__)) for name, v in found.items()
    }


@lru_cache(maxsize=1)
def ambiguous_names() -> FrozenSet[str]:
    """Names that denote more than one class, and so require a qualified name."""
    return frozenset(name for name, classes in _index().items() if len(classes) > 1)


def lookup(name: str) -> type:
    """Resolve a name to a data model class.

    Accepts a class name (``GeneticAlgorithmOptimizer``), a ``type``
    discriminator value (``GeneticAlgorithm``) or a fully qualified name.

    Raises:
        UnknownNameError: If nothing is registered under *name*.
        AmbiguousNameError: If several classes share *name*.
    """
    classes = _index().get(name)
    if not classes:
        raise UnknownNameError(f"No data model named {name!r}.")
    if len(classes) > 1:
        qualified = "\n  ".join(f"{c.__module__}.{c.__qualname__}" for c in classes)
        raise AmbiguousNameError(
            f"{name!r} is defined in {len(classes)} places. "
            f"Use a fully qualified name:\n  {qualified}"
        )
    return classes[0]


def names() -> FrozenSet[str]:
    """Every name the index resolves, including tags and qualified names."""
    return frozenset(_index())
