"""Walking a path through a data model schema, optionally alongside a spec.

A path is a list of segments: strings name a field, a union member or an
element key, integers name a list position. Without a spec the first segment
names a type and the walk is over the schema only. With a spec the path is
relative to the spec's root and the spec supplies element keys and current
values.

The walk consumes one segment at a time and never expands a subtree, so the
recursive parts of the model graph (``StepwiseStrategy -> Step ->
StepwiseStrategy``) are bounded by the path length.
"""

import json
from dataclasses import dataclass, replace
from typing import Any, Optional, Sequence, Tuple, Union

from pydantic.fields import FieldInfo

from bofire.data_models.introspection.annotations import (
    TypeNode,
    model_cls,
    normalize,
    type_tag,
)
from bofire.data_models.introspection.containers import (
    KEY_FIELD,
    Container,
    container_for,
)
from bofire.data_models.introspection.registry import lookup


__all__ = ["MISSING", "Cursor", "PathError", "resolve"]

Segment = Union[str, int]

#: The spec has no value at this location (as opposed to a value of ``None``).
MISSING = object()


class PathError(ValueError):
    """Raised when a path is malformed or does not resolve."""


@dataclass(frozen=True)
class Cursor:
    """A location reached by walking a path."""

    node: TypeNode
    path: Tuple[Segment, ...]
    field: Optional[FieldInfo] = None
    has_spec: bool = False
    value: Any = MISSING
    container: Optional[Container] = None


def _fmt(path: Sequence[Segment]) -> str:
    return json.dumps(list(path))


def _tags(node: TypeNode) -> list:
    return [type_tag(option) or option.__name__ for option in node.options]


def _model_cursor(cls: type, cursor: Cursor, path: Tuple[Segment, ...]) -> Cursor:
    return replace(
        cursor,
        node=TypeNode(kind="model", cls=cls, display=cls.__name__),
        path=path,
        field=None,
        container=container_for(cls),
    )


def _element_container(node: TypeNode) -> Optional[Container]:
    """Lists whose elements carry a ``key`` field (features) are key-addressable."""
    element = node.element
    if element is None:
        return None
    options = element.options if element.kind == "union" else (element.cls,)
    if all(
        isinstance(o, type) and "key" in getattr(o, "model_fields", {}) for o in options
    ):
        return KEY_FIELD
    return None


def _step_union(cursor: Cursor, segment: Segment) -> Cursor:
    node = cursor.node
    if cursor.has_spec:
        tag = (
            cursor.value.get(node.discriminator or "type")
            if isinstance(cursor.value, dict)
            else None
        )
        chosen = next(
            (o for o in node.options if tag in (type_tag(o), o.__name__)), None
        )
        if chosen is None:
            raise PathError(
                f"{_fmt(cursor.path)}: value has no valid {node.discriminator!r} "
                f"tag; expected one of {_tags(node)}."
            )
        return step(_model_cursor(chosen, cursor, cursor.path), segment)

    chosen = next(
        (o for o in node.options if segment in (type_tag(o), o.__name__)), None
    )
    if chosen is None:
        raise PathError(
            f"{_fmt(cursor.path)} is a union; name a member first, one of "
            f"{_tags(node)}."
        )
    return _model_cursor(chosen, cursor, cursor.path + (segment,))


def _step_sequence(cursor: Cursor, segment: Segment) -> Cursor:
    node = cursor.node
    path = cursor.path + (segment,)
    element = node.element
    value: Any = MISSING

    if isinstance(segment, int):
        if cursor.has_spec:
            items = cursor.value if isinstance(cursor.value, list) else []
            if not -len(items) <= segment < len(items):
                raise PathError(
                    f"{_fmt(path)}: index out of range ({len(items)} items)."
                )
            value = items[segment]
    else:
        if not cursor.has_spec:
            raise PathError(
                f"{_fmt(path)}: element keys are data, not schema; pass a spec."
            )
        if node.kind == "mapping":
            mapping = cursor.value if isinstance(cursor.value, dict) else {}
            if segment not in mapping:
                raise PathError(
                    f"{_fmt(path)}: no key {segment!r}. Keys: {list(mapping)}."
                )
            value = mapping[segment]
        else:
            container = cursor.container or _element_container(node)
            if container is None or not container.keyed:
                raise PathError(f"{_fmt(path)}: this list is addressed by index.")
            items = cursor.value if isinstance(cursor.value, list) else []
            keys = [container.key_of(item) for item in items]
            if segment not in keys:
                raise PathError(
                    f"{_fmt(path)}: no element keyed {segment!r}. "
                    f"Keys: {[k for k in keys if k is not None]}."
                )
            value = items[keys.index(segment)]

    return Cursor(
        node=element if element is not None else TypeNode(kind="scalar", display="Any"),
        path=path,
        has_spec=cursor.has_spec,
        value=value,
        container=container_for(element.cls) if element is not None else None,
    )


def _step_model(cursor: Cursor, segment: Segment) -> Cursor:
    cls = model_cls(cursor.node)

    # Declared fields win over element keys of the same name.
    if isinstance(segment, str) and segment in cls.model_fields:
        field = cls.model_fields[segment]
        node = normalize(field.annotation, field)
        own = cursor.container
        if own is not None and segment == own.list_field:
            container: Optional[Container] = own
        else:
            container = container_for(node.cls) or _element_container(node)
        value = (
            cursor.value.get(segment, MISSING)
            if cursor.has_spec and isinstance(cursor.value, dict)
            else MISSING
        )
        return Cursor(
            node=node,
            path=cursor.path + (segment,),
            field=field,
            has_spec=cursor.has_spec,
            value=value,
            container=container,
        )

    # Container shortcut: ["domain", "inputs", "temperature"] skips "features".
    container = cursor.container
    if container is not None and (container.keyed or isinstance(segment, int)):
        inner = cls.model_fields[container.list_field]
        list_cursor = Cursor(
            node=normalize(inner.annotation, inner),
            path=cursor.path,
            field=inner,
            has_spec=cursor.has_spec,
            value=(
                cursor.value.get(container.list_field, MISSING)
                if isinstance(cursor.value, dict)
                else MISSING
            ),
            container=container,
        )
        return _step_sequence(list_cursor, segment)

    fields = sorted(name for name in cls.model_fields if name != "type")
    hint = (
        f" Its elements are addressed by index, e.g. {_fmt(cursor.path + (0,))}."
        if container is not None and not container.keyed
        else ""
    )
    raise PathError(
        f"{_fmt(cursor.path)}: {cls.__name__} has no field {segment!r}. "
        f"Fields: {fields}.{hint}"
    )


def step(cursor: Cursor, segment: Segment) -> Cursor:
    """Consume one segment."""
    kind = cursor.node.kind
    if kind == "union":
        return _step_union(cursor, segment)
    if kind in ("list", "mapping"):
        return _step_sequence(cursor, segment)
    if kind == "model":
        return _step_model(cursor, segment)
    raise PathError(f"{_fmt(cursor.path)} is a leaf; there is nothing below it.")


def _check_path(path: Any) -> Tuple[Segment, ...]:
    if isinstance(path, (str, bytes)) or not isinstance(path, Sequence):
        raise PathError(
            f"A path is a list of segments, e.g. ['SoboStrategy', "
            f"'acquisition_function'], got {path!r}."
        )
    for segment in path:
        if isinstance(segment, bool) or not isinstance(segment, (str, int)):
            raise PathError(f"Path segments are strings or integers, got {segment!r}.")
    return tuple(path)


def resolve(
    path: Sequence[Segment],
    spec: Optional[dict] = None,
    root: Optional[str] = None,
) -> Cursor:
    """Resolve *path*, over the schema alone or alongside *spec*.

    Args:
        path: The segments to walk.
        spec: A serialized data model. When given, *path* is relative to it.
        root: The type of *spec*, needed only when it carries no ``type``.

    Raises:
        PathError: If the path is malformed or does not resolve.
        UnknownNameError: If a type name is not registered.
        AmbiguousNameError: If a type name is defined in more than one place.
    """
    segments = _check_path(path)

    if spec is not None:
        if not isinstance(spec, dict):
            raise PathError("spec must be a serialized data model (a dict).")
        name = root if root is not None else spec.get("type")
        if not isinstance(name, str):
            raise PathError("spec carries no 'type'; pass root='<TypeName>'.")
        cls = lookup(name)
        cursor = Cursor(
            node=TypeNode(kind="model", cls=cls, display=cls.__name__),
            path=(),
            has_spec=True,
            value=spec,
            container=container_for(cls),
        )
    else:
        if not segments or not isinstance(segments[0], str):
            raise PathError("Without a spec, a path must start with a type name.")
        cls = lookup(segments[0])
        cursor = Cursor(
            node=TypeNode(kind="model", cls=cls, display=cls.__name__),
            path=segments[:1],
            container=container_for(cls),
        )
        segments = segments[1:]

    for segment in segments:
        cursor = step(cursor, segment)
    return cursor
