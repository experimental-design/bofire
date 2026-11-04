"""Rendering a resolved :class:`~.resolver.Cursor` as a compact dict.

The summary lists one node's own fields, naming union members instead of
expanding them. ``full=True`` instead returns the node's JSON schema with
``$defs`` dropped and each ``$ref`` left as a bare type name, which ``describe``
accepts, so its size depends on the node alone.

Keys with nothing to report are omitted rather than set to ``None``.
"""

import json
from typing import Any, Dict, Optional, Type

from pydantic import BaseModel
from pydantic_core import PydanticUndefined

from bofire.data_models.introspection.annotations import (
    TypeNode,
    enum_cls,
    model_cls,
    normalize,
    type_tag,
)
from bofire.data_models.introspection.resolver import MISSING, Cursor


__all__ = ["approx_tokens", "render", "shallow_schema"]

#: Above this many union members only their count is listed.
MAX_INLINE_OPTIONS = 12
#: Model-valued defaults whose JSON is longer than this are shown by tag only.
MAX_INLINE_DEFAULT_CHARS = 120


def approx_tokens(payload: Any) -> int:
    """Approximate token count of a payload: JSON byte length over four."""
    return len(json.dumps(payload, separators=(",", ":"), default=str)) // 4


def _tag(cls: type) -> str:
    return type_tag(cls) or cls.__name__


def _union_of(node: TypeNode) -> Optional[TypeNode]:
    if node.kind == "union":
        return node
    if node.element is not None and node.element.kind == "union":
        return node.element
    return None


def _options(union: TypeNode) -> Dict[str, Any]:
    tags = [_tag(option) for option in union.options]
    if len(tags) <= MAX_INLINE_OPTIONS:
        return {"options": tags}
    return {"options_count": len(tags)}


def _is(node: TypeNode, value: Any) -> Optional[str]:
    """The tag of the union member *value* holds, if it holds one."""
    if node.kind != "union" or not isinstance(value, dict):
        return None
    tag = value.get(node.discriminator or "type")
    return tag if any(tag in (type_tag(o), o.__name__) for o in node.options) else None


def _default(field: Any, node: TypeNode) -> Any:
    """The field's default, including one declared via ``default_factory``.

    Pydantic omits ``default_factory`` values from JSON schema.
    """
    if field is None:
        return PydanticUndefined
    if field.default_factory is not None:
        try:
            default = field.default_factory()
        except TypeError:  # factory needs validated data
            return PydanticUndefined
    else:
        default = field.default
    if isinstance(default, BaseModel):
        dumped = default.model_dump(mode="json")
        if node.kind == "union" or len(json.dumps(dumped)) > MAX_INLINE_DEFAULT_CHARS:
            return _tag(type(default))
        return dumped
    if default is PydanticUndefined:
        return default
    return json.loads(json.dumps(default, default=str))


def _describe_field(field: Any, node: TypeNode, cursor_value: Any) -> Dict[str, Any]:
    """The metadata shared by a class's field entry and a leaf node."""
    entry: Dict[str, Any] = {"type": node.display}
    union = _union_of(node)
    if union is not None:
        entry.update(_options(union))
        if "options" in entry:  # the members are listed; don't repeat them
            entry["type"] = ("union" if node.kind == "union" else "list[union]") + (
                "?" if node.optional else ""
            )
    default = _default(field, node)
    if default is not PydanticUndefined:
        entry["default"] = default
    elif field is not None and field.is_required():
        entry["required"] = True
    if field is not None and field.description:
        entry["description"] = field.description
    if cursor_value is not MISSING:
        if node.kind == "union":
            tag = _is(node, cursor_value)
            if tag is not None:
                entry["is"] = tag
        elif not _navigable(node):
            entry["value"] = cursor_value
    return entry


def _navigable(node: TypeNode) -> bool:
    """Whether a node is descended into rather than read as a value."""
    if node.kind in ("model", "union"):
        return True
    return (
        node.kind in ("list", "mapping")
        and node.element is not None
        and (node.element.kind in ("model", "union"))
    )


def _render_class(cursor: Cursor) -> Dict[str, Any]:
    cls = model_cls(cursor.node)
    payload: Dict[str, Any] = {"kind": "class", "name": cls.__name__}
    if _tag(cls) != cls.__name__:
        payload["tag"] = _tag(cls)
    doc = (cls.__dict__.get("__doc__") or "").strip().split("\n\n")[0]
    if doc:
        payload["doc"] = " ".join(doc.split())
    value = cursor.value if isinstance(cursor.value, dict) else {}
    payload["fields"] = {
        name: _describe_field(
            field,
            normalize(field.annotation, field),
            value.get(name, MISSING) if cursor.has_spec else MISSING,
        )
        for name, field in cls.model_fields.items()
        if name != "type"
    }
    return payload


def _render_union(cursor: Cursor) -> Dict[str, Any]:
    node = cursor.node
    payload: Dict[str, Any] = {"kind": "union", "discriminator": node.discriminator}
    payload.update(_options(node))
    if node.optional:
        payload["optional"] = True
    if node.abstract:
        payload["abstract_bases"] = [base.__name__ for base in node.abstract]
    if cursor.has_spec:
        tag = _is(node, cursor.value)
        if tag is not None:
            payload["is"] = tag
    return payload


def _render_list(cursor: Cursor) -> Dict[str, Any]:
    """A list or mapping of models, or a container model such as ``Inputs``."""
    node, container, items = cursor.node, cursor.container, cursor.value
    if node.kind == "model" and container is not None:
        inner = model_cls(node).model_fields[container.list_field]
        element = normalize(inner.annotation, inner).element
        if isinstance(items, dict):
            items = items.get(container.list_field, [])
    else:
        element = node.element

    payload: Dict[str, Any] = {"kind": "list"}
    if node.kind == "model":
        payload["name"] = node.display
    if element is not None:
        union = element if element.kind == "union" else None
        if union is not None:
            payload.update(_options(union))
        else:
            payload["element"] = element.display
    keyed = container is not None and container.keyed
    if node.kind == "mapping":
        payload["addressing"] = "key"
        if cursor.has_spec and isinstance(cursor.value, dict):
            payload["keys"] = list(cursor.value)
        return payload
    payload["addressing"] = "key" if keyed else "index"
    if cursor.has_spec and isinstance(items, list):
        if keyed and container is not None:
            payload["keys"] = {
                key: (_is(element, item) if element is not None else None) or "?"
                for item in items
                if (key := container.key_of(item)) is not None
            }
        else:
            payload["length"] = len(items)
    return payload


def _render_leaf(cursor: Cursor) -> Dict[str, Any]:
    node = cursor.node
    payload: Dict[str, Any] = {"kind": "leaf"}
    payload.update(_describe_field(cursor.field, node, cursor.value))
    if node.kind == "enum":
        payload["values"] = [member.value for member in enum_cls(node)]
    if node.kind == "literal":
        payload["const"] = node.literal_values[0]
    return payload


def shallow_schema(cls: Type[BaseModel]) -> Dict[str, Any]:
    """A model's JSON schema with ``$defs`` dropped and refs left as type names."""
    schema = cls.model_json_schema(ref_template="{model}")
    schema.pop("$defs", None)
    schema["refs_are_names"] = True
    return schema


def render(cursor: Cursor, full: bool = False) -> Dict[str, Any]:
    """Render a cursor as the dict ``describe`` returns."""
    node = cursor.node
    head: Dict[str, Any] = {"path": list(cursor.path)}

    if full:
        if node.kind != "model":
            raise ValueError(
                f"{head['path']} is a {node.kind}, which has no schema of its own; "
                f"describe one of its members instead."
            )
        return {**head, **shallow_schema(model_cls(node))}

    if node.kind == "model":
        body = _render_list(cursor) if cursor.container else _render_class(cursor)
    elif node.kind == "union":
        body = _render_union(cursor)
    elif _navigable(node):
        body = _render_list(cursor)
    else:
        body = _render_leaf(cursor)
    return {**head, **body}
