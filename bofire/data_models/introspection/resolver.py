"""Walking a path through a data model, one segment at a time.

The walk is strictly lazy: :func:`step` consumes a single segment and never
expands a subtree. That is what makes the recursive parts of the model graph
(``StepwiseStrategy -> Step -> AnyStrategy -> StepwiseStrategy``,
``AdditiveKernel -> AnyKernel -> AdditiveKernel``) harmless -- a walk is bounded
by the length of the path, not by the size of the graph.

Two modes share the same code. In *schema mode* there is no instance, so a union
cannot be crossed without naming a member and element keys are unavailable: they
are data, not schema. In *instance mode* the instance collapses unions and
supplies keys, and every path is instance-relative.
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from bofire.data_models.introspection.annotations import (
    TypeNode,
    model_cls,
    normalize,
    type_tag,
)
from bofire.data_models.introspection.containers import ContainerAdapter, adapter_for
from bofire.data_models.introspection.paths import (
    PathError,
    PathSegment,
    parse_path,
    unparse,
)
from bofire.data_models.introspection.registry import lookup
from bofire.data_models.introspection.views import View


__all__ = ["Cursor", "resolve", "step"]


@dataclass(frozen=True)
class Cursor:
    """A location reached by walking a path."""

    node: TypeNode
    field: Optional[FieldInfo] = None
    view: Optional[View] = None
    segments: Tuple[PathSegment, ...] = ()
    adapter: Optional[ContainerAdapter] = None

    @property
    def path(self) -> str:
        return unparse(self.segments)

    @property
    def schema_mode(self) -> bool:
        return self.view is None


def _tags(node: TypeNode) -> List[str]:
    return [type_tag(option) or option.__name__ for option in node.options]


def _match_option(node: TypeNode, name: Any) -> Optional[type]:
    """Find the union member denoted by *name*, by tag or by class name."""
    for option in node.options:
        if name in (type_tag(option), option.__name__):
            return option
    return None


def _model_cursor(cls: type, view: Optional[View], segments) -> Cursor:
    node = TypeNode(kind="model", cls=cls, display=cls.__name__)
    return Cursor(node=node, view=view, segments=segments, adapter=adapter_for(cls))


def _did_you_mean(cursor: Cursor, segment: PathSegment) -> str:
    """Build an error naming what *is* addressable here."""
    cls = cursor.node.cls
    fields = (
        [name for name in model_cls(cursor.node).model_fields if name != "type"]
        if cls is not None
        else []
    )
    message = (
        f"{cursor.path or '<root>'}: no field {segment.value!r} on "
        f"{cls.__name__ if cls else cursor.node.display}."
    )
    if fields:
        message += f" Fields: {', '.join(sorted(fields))}."
    adapter = cursor.adapter
    if adapter is None:
        return message
    if adapter.keyed:
        if cursor.view is not None:
            keys = cursor.view.field(adapter.list_field).keys(adapter)
            if keys:
                message += f" Keys: {', '.join(keys)}."
    else:
        message += (
            f" Its elements have no key, so they are index-addressed: "
            f"{cursor.path}[0]."
        )
    return message


def _step_union(cursor: Cursor, segment: PathSegment) -> Cursor:
    node = cursor.node
    if cursor.view is not None:
        concrete = cursor.view.concrete_cls(node)
        if concrete is None:
            raise PathError(
                f"{cursor.path or '<root>'}: value carries no "
                f"{node.discriminator!r}, so it is not known which of "
                f"{_tags(node)} it is."
            )
        return step(
            _model_cursor(concrete, cursor.view, cursor.segments),
            segment,
        )

    chosen = _match_option(node, segment.value)
    if chosen is None:
        tags = _tags(node)
        raise PathError(
            f"{cursor.path or '<root>'} is a union; name one of {tags} first, "
            f"e.g. {cursor.path}.{tags[0]}.{segment.value}"
        )
    return _model_cursor(chosen, None, cursor.segments + (segment,))


def _step_sequence(cursor: Cursor, segment: PathSegment) -> Cursor:
    """Step into a list or mapping element, by index or by key."""
    node = cursor.node
    view = cursor.view
    segments = cursor.segments + (segment,)

    if segment.is_index:
        element_view = view.element(int(segment.value)) if view is not None else None
        if element_view is not None and element_view.is_missing:
            raise PathError(f"{unparse(segments)}: index out of range.")
        return Cursor(
            node=node.element,
            view=element_view,
            segments=segments,
            adapter=adapter_for(node.element.cls) if node.element else None,
        )

    if view is None:
        raise PathError(
            f"{unparse(segments)}: element keys are data, not schema. "
            f"Pass instance=..., or address a member's schema with "
            f"{cursor.path}.<TypeName>."
        )

    key = str(segment.value)
    if node.kind == "mapping":
        element_view = view.field(key)
        if element_view.is_missing:
            available = list(view.obj) if isinstance(view.obj, dict) else []
            raise PathError(
                f"{unparse(segments)}: no key {key!r}. "
                f"Keys: {', '.join(map(str, available)) or '<none>'}."
            )
    else:
        adapter = cursor.adapter or _element_adapter(node)
        if adapter is None:
            raise PathError(
                f"{unparse(segments)}: this list is index-addressed; "
                f"use {cursor.path}[0]."
            )
        index = view.index_of_key(adapter, key)
        if index is None:
            raise PathError(
                f"{unparse(segments)}: no element keyed {key!r}. "
                f"Keys: {', '.join(view.keys(adapter)) or '<none>'}."
            )
        element_view = view.element(index)

    return Cursor(
        node=node.element,
        view=element_view,
        segments=segments,
        adapter=adapter_for(node.element.cls) if node.element else None,
    )


def _element_adapter(node: TypeNode) -> Optional[ContainerAdapter]:
    """The adapter describing how this list's elements are keyed, if they are."""
    element = node.element
    if element is None:
        return None
    if element.kind == "model" and element.cls is not None:
        return _KEY_BY_KEY if _has_key_field(element.cls) else None
    if element.kind == "union" and all(map(_has_key_field, element.options)):
        return _KEY_BY_KEY
    return None


def _has_key_field(cls: type) -> bool:
    """True if instances of *cls* carry their own ``key``, as every feature does."""
    return issubclass(cls, BaseModel) and "key" in cls.model_fields


#: Elements that carry their own ``key`` field, e.g. every feature.
_KEY_BY_KEY = ContainerAdapter(list_field="", keyed=True, key_path=("key",))


def _step_model(cursor: Cursor, segment: PathSegment) -> Cursor:
    cls = model_cls(cursor.node)
    name = str(segment.value)

    # Declared fields win over element keys; see the collision rule in paths.py.
    if not segment.is_index and name in cls.model_fields:
        field = cls.model_fields[name]
        node = normalize(field.annotation, field)
        # Stepping explicitly onto a container's own list field keeps that
        # container's key semantics, so the long form of a keyed path resolves
        # exactly like the short one: ``surrogate_specs.surrogates.yield`` works
        # because ``surrogate_specs.yield`` does.
        own = adapter_for(cls)
        if own is not None and name == own.list_field:
            adapter = own
        else:
            adapter = adapter_for(node.cls) if node.cls else _element_adapter(node)
        return Cursor(
            node=node,
            field=field,
            view=cursor.view.field(name) if cursor.view is not None else None,
            segments=cursor.segments + (segment,),
            adapter=adapter,
        )

    # Container shortcut: ``domain.inputs.temperature`` and ``domain.constraints[0]``
    # both skip the wrapped list field. A name is only a shortcut when the
    # elements are keyed; an index always is.
    adapter = cursor.adapter
    if adapter is not None and (adapter.keyed or segment.is_index):
        inner = cls.model_fields[adapter.list_field]
        list_cursor = Cursor(
            node=normalize(inner.annotation, inner),
            field=inner,
            view=(
                cursor.view.field(adapter.list_field)
                if cursor.view is not None
                else None
            ),
            segments=cursor.segments,
            adapter=adapter,
        )
        return _step_sequence(list_cursor, segment)

    raise PathError(_did_you_mean(cursor, segment))


def step(cursor: Cursor, segment: PathSegment) -> Cursor:
    """Consume one segment, returning the cursor it leads to."""
    kind = cursor.node.kind
    if kind == "union":
        return _step_union(cursor, segment)
    if kind in ("list", "mapping"):
        return _step_sequence(cursor, segment)
    if kind == "model":
        return _step_model(cursor, segment)
    raise PathError(
        f"{cursor.path or '<root>'} is a {cursor.node.display} leaf; "
        f"there is nothing below it."
    )


def _root(instance: Any, root: Optional[str]) -> Cursor:
    """Build the cursor for an instance-mode root."""
    if root is not None:
        cls = lookup(root)
    elif isinstance(instance, BaseModel):
        cls = type(instance)
    elif isinstance(instance, dict) and isinstance(instance.get("type"), str):
        cls = lookup(instance["type"])
    else:
        raise PathError(
            "Cannot tell what this instance is: pass a pydantic model, a dict "
            "carrying a 'type' discriminator, or root='<TypeName>'."
        )
    return _model_cursor(cls, View(instance), ())


def resolve(
    path: str,
    instance: Any = None,
    root: Optional[str] = None,
) -> Cursor:
    """Resolve *path* to a cursor.

    Args:
        path: The path to resolve. ``""`` is the root.
        instance: A model object or serialized spec. When given, the whole path
            is instance-relative; otherwise the first segment names a type.
        root: The type name of *instance*, needed only for a dict that carries
            no ``type`` discriminator.

    Raises:
        PathError: If the path is malformed or does not resolve.
        UnknownNameError: If a type name is not registered.
    """
    segments = parse_path(path)

    if instance is not None:
        cursor = _root(instance, root)
    else:
        if not segments:
            raise PathError(
                "Empty path with no instance: pass a type name such as "
                "'SoboStrategy', or an instance."
            )
        head, segments = segments[0], segments[1:]
        if head.is_index:
            raise PathError(f"{path!r}: a path must start with a type name.")
        cursor = _model_cursor(lookup(str(head.value)), None, (head,))

    for segment in segments:
        cursor = step(cursor, segment)
    return cursor
