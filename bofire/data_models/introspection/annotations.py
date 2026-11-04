"""Normalisation of pydantic field annotations into a uniform :class:`TypeNode`.

Annotations in ``bofire.data_models`` appear in several shapes that must all be
reduced to the same structure before a path can be walked through them:

* a bare tagged union -- ``SoboStrategy.acquisition_function``
* an optional tagged union -- ``SoboStrategy.convergence_criterion``
* a sequence of a tagged union -- ``Inputs.features``
* a ``TypeVar`` standing in for one -- ``Constraints.constraints``

The last two shapes carry their discriminator on the *element*, and the optional
shape hides it from ``FieldInfo.discriminator`` entirely: pydantic does not hoist
a discriminator through ``Optional[...]``, so ``FieldInfo.discriminator`` is
``None`` for ``convergence_criterion`` even though it is a tagged union. Reading
that attribute alone therefore renders several unions as untagged, which is why
:func:`normalize` -- not ``unions.extract_union_args`` -- is the entry point.
"""

import collections.abc
import enum
from dataclasses import dataclass, replace
from typing import (
    Any,
    Literal,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from bofire.data_models._register_utils import _type_of
from bofire.data_models.unions import discriminator_name, unwrap_annotated


__all__ = ["TypeNode", "enum_cls", "model_cls", "normalize", "type_tag"]

NodeKind = Literal["model", "union", "list", "mapping", "enum", "literal", "scalar"]

_SEQUENCE_ORIGINS = (
    list,
    set,
    frozenset,
    collections.abc.Sequence,
    collections.abc.Set,
)
_MAPPING_ORIGINS = (dict, collections.abc.Mapping)


@dataclass(frozen=True)
class TypeNode:
    """A pydantic annotation reduced to the shape a path walker needs.

    ``kind`` selects which of the remaining attributes are meaningful:
    ``model`` uses ``cls``; ``union`` uses ``options`` / ``abstract`` /
    ``discriminator``; ``list`` and ``mapping`` use ``element``; ``enum`` uses
    ``cls``; ``literal`` uses ``literal_values``.
    """

    kind: NodeKind
    cls: Optional[type] = None
    options: Tuple[type, ...] = ()
    abstract: Tuple[type, ...] = ()
    discriminator: Optional[str] = None
    element: Optional["TypeNode"] = None
    key_type: Optional[type] = None
    optional: bool = False
    literal_values: Tuple[Any, ...] = ()
    display: str = ""


#: Stands in for an element type that a bare container annotation does not give.
_ANY = TypeNode(kind="scalar", display="Any")


def model_cls(node: TypeNode) -> Type[BaseModel]:
    """Narrow a node's class to a pydantic model.

    ``TypeNode.cls`` holds whatever the node's ``kind`` implies -- a model, an
    enum, or a plain type -- so callers that have already established the kind
    use this to say so once rather than asserting at every attribute access.
    """
    cls = node.cls
    if cls is None or not (isinstance(cls, type) and issubclass(cls, BaseModel)):
        raise TypeError(f"{node.display or node.kind} is not a model node.")
    return cls


def enum_cls(node: TypeNode) -> Type[enum.Enum]:
    """Narrow a node's class to an enum, for nodes of kind ``enum``."""
    cls = node.cls
    if cls is None or not (isinstance(cls, type) and issubclass(cls, enum.Enum)):
        raise TypeError(f"{node.display or node.kind} is not an enum node.")
    return cls


def type_tag(cls: Any) -> Optional[str]:
    """Return a class's ``type`` discriminator value, or ``None`` if it has none.

    Abstract bases such as ``Constraint`` declare ``type: Any`` and concrete
    containers such as ``BotorchSurrogates`` declare no ``type`` at all; both
    yield ``None``. The tag is what a caller writes into JSON, and it is not
    always the class name -- ``GeneticAlgorithmOptimizer`` has the tag
    ``"GeneticAlgorithm"``.
    """
    if not (isinstance(cls, type) and issubclass(cls, BaseModel)):
        return None
    try:
        return _type_of(cls, "type")
    except ValueError:
        return None


def _display(annotation: Any) -> str:
    """Render an annotation as a short human- and LLM-readable type string."""
    annotation, _ = unwrap_annotated(annotation)
    if annotation is type(None):
        return "None"
    origin = get_origin(annotation)
    if origin is None:
        return getattr(annotation, "__name__", str(annotation))
    args = [_display(a) for a in get_args(annotation) if a is not type(None)]
    if origin is Union:
        return " | ".join(args)
    if origin is Literal:
        return "Literal[" + ", ".join(repr(a) for a in get_args(annotation)) + "]"
    name = getattr(origin, "__name__", str(origin))
    return f"{name}[{', '.join(args)}]" if args else name


def _union_display(options: Tuple[type, ...]) -> str:
    return "union[" + "|".join(type_tag(o) or o.__name__ for o in options) + "]"


def _normalize_union(
    annotation: Any,
    discriminator: Optional[str],
    seen: frozenset,
) -> TypeNode:
    """Handle ``Union[...]``, including the optional and nested-union shapes."""
    raw = get_args(annotation)
    optional = type(None) in raw
    rest = [a for a in raw if a is not type(None)]

    # ``Optional[X]`` with a single survivor: recurse so that a discriminator
    # carried by an inner ``Annotated`` is recovered rather than lost.
    if len(rest) == 1:
        inner = normalize(rest[0], None, seen)
        return replace(
            inner,
            optional=True,
            discriminator=inner.discriminator or discriminator,
            display=inner.display + "?",
        )

    # Flatten one level of nesting. ``Constraints`` reaches us as
    # ``Union[Annotated[Union[...concrete...], Field(discriminator)], Constraint]``.
    flat = []
    for arg in rest:
        inner, meta = unwrap_annotated(arg)
        discriminator = discriminator or discriminator_name(meta)
        if get_origin(inner) is Union:
            flat.extend(get_args(inner))
        else:
            flat.append(arg)

    members = [unwrap_annotated(a)[0] for a in flat]
    concrete: list = []
    abstract: list = []
    for member in members:
        if not (isinstance(member, type) and issubclass(member, BaseModel)):
            concrete = []
            break
        (concrete if type_tag(member) else abstract).append(member)

    if len(concrete) >= 2:
        options = tuple(concrete)
        return TypeNode(
            kind="union",
            options=options,
            abstract=tuple(abstract),
            discriminator=discriminator or "type",
            optional=optional,
            display=_union_display(options),
        )

    # A union of non-models, e.g. ``ContinuousInput.bounds`` which is
    # ``Union[List[float], Tuple[float, float]]``. There is nothing to navigate.
    return TypeNode(
        kind="scalar",
        optional=optional,
        display=" | ".join(_display(a) for a in rest),
    )


def normalize(
    annotation: Any,
    field: Optional[FieldInfo] = None,
    seen: frozenset = frozenset(),
) -> TypeNode:
    """Reduce a pydantic annotation to a :class:`TypeNode`.

    Args:
        annotation: The annotation to normalise.
        field: The ``FieldInfo`` it came from, if any. Used only as a fallback
            source for the discriminator.
        seen: ``TypeVar``s already resolved on this branch, guarding against a
            self-bounded type variable.

    Returns:
        The normalised node.
    """
    annotation, meta = unwrap_annotated(annotation)
    discriminator = discriminator_name(meta) or (
        field.discriminator if field is not None else None
    )

    # ``Constraints.constraints`` is ``Sequence[~C]``: the container is generic
    # and used unparametrized, so the options live on the TypeVar's bound.
    if isinstance(annotation, TypeVar):
        if annotation in seen:
            return TypeNode(kind="scalar", display="Any")
        bound = annotation.__bound__ or object
        return normalize(bound, field, seen | {annotation})

    origin = get_origin(annotation)

    if origin is Union:
        return _normalize_union(annotation, discriminator, seen)

    if origin in _SEQUENCE_ORIGINS or _is_variadic_tuple(annotation, origin):
        # A bare ``List``/``Sequence`` carries no element type.
        args = get_args(annotation)
        element = normalize(args[0], None, seen) if args else _ANY
        return TypeNode(
            kind="list",
            element=element,
            discriminator=element.discriminator,
            display=f"list[{element.display}]",
        )

    if origin in _MAPPING_ORIGINS:
        args = get_args(annotation)
        key_type = args[0] if len(args) == 2 else None
        element = normalize(args[1], None, seen) if len(args) == 2 else _ANY
        key_display = _display(key_type) if key_type is not None else "Any"
        return TypeNode(
            kind="mapping",
            key_type=key_type,
            element=element,
            discriminator=element.discriminator,
            display=f"dict[{key_display}, {element.display}]",
        )

    if origin is Literal:
        values = get_args(annotation)
        return TypeNode(
            kind="literal",
            literal_values=values,
            display=f"const {values[0]!r}"
            if len(values) == 1
            else _display(annotation),
        )

    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        return TypeNode(kind="enum", cls=annotation, display=annotation.__name__)

    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return TypeNode(kind="model", cls=annotation, display=annotation.__name__)

    return TypeNode(
        kind="scalar",
        cls=annotation if isinstance(annotation, type) else None,
        display=_display(annotation),
    )


def _is_variadic_tuple(annotation: Any, origin: Any) -> bool:
    """True for ``Tuple[X, ...]``, which is a sequence rather than a value.

    A fixed-length tuple such as ``Tuple[float, float]`` (``ContinuousInput.bounds``)
    is a value and must stay a scalar.
    """
    if origin is not tuple:
        return False
    args = get_args(annotation)
    return len(args) == 2 and args[1] is Ellipsis
