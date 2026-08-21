"""Rendering a resolved :class:`~.resolver.Cursor` as a compact dict.

Two shapes are produced. The *summary* is what makes progressive disclosure
work: one node's own fields, with union-typed fields listed as the names of
their members rather than expanded. The *full* form is the node's complete JSON
schema with ``$defs`` stripped and every ``$ref`` rewritten to a bare type name
-- which is itself a valid argument to ``describe``.

Stripping ``$defs`` is not cosmetic. ``SingleTaskGPSurrogate.model_json_schema()``
is ~25,800 tokens and ``BotorchSurrogates`` ~35,900, almost entirely transitive
definitions; the shallow form is ~944 and ~446. The cost of the shallow form is
bounded by the node's own field count, never by the size of the graph, which is
also why it is safe on the model graph's cycles.

Keys are omitted when they have nothing to say, never emitted as ``null``: most
data models are still undocumented, and an omitted ``description`` costs nothing
while ``"description": null`` costs tokens on every field.
"""

import json
from dataclasses import replace
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel
from pydantic_core import PydanticUndefined

from bofire.data_models.introspection.annotations import (
    TypeNode,
    enum_cls,
    model_cls,
    normalize,
    type_tag,
)
from bofire.data_models.introspection.containers import adapter_for
from bofire.data_models.introspection.resolver import Cursor
from bofire.data_models.introspection.views import View


__all__ = ["approx_tokens", "render", "shallow_schema"]

#: Above this many union members, list the count and let the agent drill in.
MAX_INLINE_OPTIONS = 12
#: Defaults whose JSON is longer than this are summarised rather than inlined.
MAX_INLINE_DEFAULT_CHARS = 120
#: Docstrings are truncated to this many characters in the summary form.
MAX_DOC_CHARS = 240


def approx_tokens(payload: Any) -> int:
    """Estimate the token cost of a payload.

    A byte count over four, which is close enough for budgeting and needs no
    tokenizer dependency -- the data-model test job installs only the package
    and pytest.
    """
    return len(json.dumps(payload, separators=(",", ":"), default=str)) // 4


def _doc(cls: type) -> Optional[str]:
    """The first paragraph of a class's own docstring, truncated."""
    raw = cls.__dict__.get("__doc__")
    if not raw:
        return None
    paragraph = " ".join(raw.strip().split("\n\n")[0].split())
    if not paragraph:
        return None
    if len(paragraph) > MAX_DOC_CHARS:
        paragraph = paragraph[: MAX_DOC_CHARS - 1].rstrip() + "…"
    return paragraph


def _tags(node: TypeNode) -> List[str]:
    return [type_tag(option) or option.__name__ for option in node.options]


def _default_of(field: Any) -> Any:
    """The field's default, resolving ``default_factory``.

    Pydantic omits ``default_factory`` values from JSON schema entirely, so an
    agent reading the raw schema cannot see the default for fields such as
    ``acquisition_function``. Resolving it here closes that gap for navigation.
    """
    if field is None:
        return PydanticUndefined
    if field.default_factory is not None:
        try:
            return field.default_factory()
        except TypeError:
            # Factories taking validated data cannot be resolved out of context.
            return PydanticUndefined
    return field.default


def _render_default(field: Any, node: TypeNode) -> Any:
    """Render a default compactly: a tag for unions, a value for scalars."""
    default = _default_of(field)
    if default is PydanticUndefined:
        return PydanticUndefined
    if isinstance(default, BaseModel):
        tag = type_tag(type(default))
        if node.kind == "union":
            return tag or type(default).__name__
        dumped = default.model_dump()
        if len(json.dumps(dumped, default=str)) <= MAX_INLINE_DEFAULT_CHARS:
            return dumped
        return tag or type(default).__name__
    return View(default).json_value()


def _field_entry(name: str, field: Any, view: Optional[View]) -> Dict[str, Any]:
    """One line of a class summary."""
    node = normalize(field.annotation, field)
    entry: Dict[str, Any] = {"type": node.display}

    if node.kind == "union" or (node.element and node.element.kind == "union"):
        union = node if node.kind == "union" else node.element
        tags = _tags(union)
        if len(tags) <= MAX_INLINE_OPTIONS:
            entry["options"] = tags
            # ``options`` already names the members; spelling them out again in
            # ``type`` would double the cost of the commonest kind of field.
            entry["type"] = ("union" if node.kind == "union" else "list[union]") + (
                "?" if node.optional else ""
            )
        else:
            entry["options_count"] = len(tags)

    default = _render_default(field, node)
    if default is not PydanticUndefined:
        entry["default"] = default
    elif field.is_required():
        entry["required"] = True

    if field.description:
        entry["description"] = field.description

    if view is not None:
        member = view.field(name)
        if member.is_missing:
            entry["present"] = False
        elif node.kind == "union":
            concrete = member.concrete_cls(node)
            if concrete is not None:
                entry["is"] = type_tag(concrete) or concrete.__name__
        elif node.kind in ("model", "list", "mapping"):
            pass  # navigable; the agent descends rather than reading it here
        else:
            entry["value"] = member.json_value()
    return entry


def _render_class(cursor: Cursor) -> Dict[str, Any]:
    cls = model_cls(cursor.node)
    payload: Dict[str, Any] = {"kind": "class", "name": cls.__name__}
    tag = type_tag(cls)
    if tag and tag != cls.__name__:
        payload["tag"] = tag
    doc = _doc(cls)
    if doc:
        payload["doc"] = doc
    # ``type`` is the discriminator and is implied by ``name``/``tag``.
    payload["fields"] = {
        name: _field_entry(name, field, cursor.view)
        for name, field in cls.model_fields.items()
        if name != "type"
    }
    return payload


def _render_union(cursor: Cursor) -> Dict[str, Any]:
    node = cursor.node
    payload: Dict[str, Any] = {
        "kind": "union",
        "discriminator": node.discriminator,
        "options": _tags(node),
    }
    if node.optional:
        payload["optional"] = True
    if node.abstract:
        payload["abstract_bases"] = [base.__name__ for base in node.abstract]
    if cursor.view is not None:
        concrete = cursor.view.concrete_cls(node)
        if concrete is not None:
            payload["is"] = type_tag(concrete) or concrete.__name__
    return payload


def _render_container(cursor: Cursor) -> Dict[str, Any]:
    """A container model: keyed like ``Inputs`` or index-addressed like ``Constraints``."""
    cls = model_cls(cursor.node)
    adapter = cursor.adapter
    assert adapter is not None
    inner = cls.model_fields[adapter.list_field]
    element = normalize(inner.annotation, inner).element

    payload: Dict[str, Any] = {
        "kind": "keyed_container" if adapter.keyed else "list",
        "name": cls.__name__,
        "addressing": adapter.addressing,
        "list_field": adapter.list_field,
    }
    if element is not None and element.kind == "union":
        tags = _tags(element)
        if len(tags) <= MAX_INLINE_OPTIONS:
            payload["options"] = tags
        else:
            payload["options_count"] = len(tags)
        if element.abstract:
            payload["abstract_bases"] = [b.__name__ for b in element.abstract]

    if cursor.view is None:
        payload["keys" if adapter.keyed else "length"] = "<requires instance>"
        return payload

    list_view = cursor.view.field(adapter.list_field)
    if adapter.keyed:
        keys: Dict[str, str] = {}
        for item in list_view.items():
            key = adapter.key_of(item)
            if key is None:
                continue
            item_view = View(item)
            concrete = item_view.concrete_cls(element) if element else None
            keys[key] = (type_tag(concrete) or concrete.__name__) if concrete else "?"
        payload["keys"] = keys
        shadowed = [key for key in keys if key in cls.model_fields]
        if shadowed:
            payload["shadowed_keys"] = shadowed
    else:
        payload["length"] = len(list_view.items())
        payload["items"] = [
            _summarise_element(index, item, element)
            for index, item in enumerate(list_view.items())
        ]
    return payload


def _summarise_element(
    index: int, item: Any, element: Optional[TypeNode]
) -> Dict[str, Any]:
    entry: Dict[str, Any] = {"index": index}
    concrete = View(item).concrete_cls(element) if element else None
    if concrete is not None:
        entry["is"] = type_tag(concrete) or concrete.__name__
    summary = _summary_text(item, concrete)
    if summary is not None:
        entry["summary"] = summary
    return entry


def _summary_text(item: Any, concrete: Optional[type]) -> Optional[str]:
    """One-line prose for a list element, if the element type provides it.

    ``to_description`` is an existing convention across constraints, features and
    objectives, so reuse it rather than inventing a second prose channel. A
    serialized element is revived first, so that describing a spec dict yields
    the same output as describing the object it came from.
    """
    if isinstance(item, dict):
        if concrete is None:
            return None
        try:
            item = concrete(**item)
        except Exception:  # noqa: BLE001 - a partial element simply has no summary
            return None
    render_it = getattr(item, "to_description", None)
    if not callable(render_it):
        return None
    try:
        return render_it()
    except Exception:  # noqa: BLE001 - a summary must never break navigation
        return None


def _render_sequence(cursor: Cursor) -> Dict[str, Any]:
    node = cursor.node
    payload: Dict[str, Any] = {"kind": node.kind, "type": node.display}
    element = node.element
    if element is not None and element.kind == "union":
        tags = _tags(element)
        if len(tags) <= MAX_INLINE_OPTIONS:
            payload["options"] = tags
        else:
            payload["options_count"] = len(tags)
    if node.kind == "mapping" and node.key_type is not None:
        payload["key_type"] = getattr(node.key_type, "__name__", str(node.key_type))
    if cursor.view is not None:
        if node.kind == "mapping":
            payload["keys"] = (
                list(cursor.view.obj) if isinstance(cursor.view.obj, dict) else []
            )
        else:
            payload["length"] = len(cursor.view.items())
    return payload


def _render_leaf(cursor: Cursor) -> Dict[str, Any]:
    node = cursor.node
    # A plain value renders as ``leaf``; enums and literals keep their own kind
    # because each carries extra information the caller can act on.
    payload: Dict[str, Any] = {
        "kind": "leaf" if node.kind == "scalar" else node.kind,
        "type": node.display,
    }
    if node.kind == "enum":
        payload["values"] = [member.value for member in enum_cls(node)]
    if node.kind == "literal":
        payload["const"] = node.literal_values[0] if node.literal_values else None
    field = cursor.field
    if field is not None:
        if field.description:
            payload["description"] = field.description
        default = _render_default(field, node)
        if default is not PydanticUndefined:
            payload["default"] = default
        elif field.is_required():
            payload["required"] = True
        constraints = _constraints_of(field)
        if constraints:
            payload["constraints"] = constraints
    if cursor.view is not None:
        if cursor.view.is_missing:
            payload["present"] = False
        else:
            payload["value"] = cursor.view.json_value()
    return payload


def _constraints_of(field: Any) -> Dict[str, Any]:
    """Render validation metadata (``Ge``, ``MinLen``, ...) attached to a field."""
    constraints: Dict[str, Any] = {}
    for item in getattr(field, "metadata", ()):  # annotated-types instances
        for attribute in ("ge", "gt", "le", "lt", "min_length", "max_length"):
            value = getattr(item, attribute, None)
            if value is not None:
                constraints[attribute] = value
    return constraints


def _render_concrete(cursor: Cursor, concrete: type) -> Dict[str, Any]:
    """Render the union member an instance actually holds."""
    resolved = replace(
        cursor,
        node=TypeNode(kind="model", cls=concrete, display=concrete.__name__),
        adapter=adapter_for(concrete),
    )
    if resolved.adapter is not None:
        return _render_container(resolved)
    return _render_class(resolved)


def shallow_schema(cls: Type[BaseModel]) -> Dict[str, Any]:
    """A model's JSON schema with ``$defs`` dropped and refs left as type names."""
    schema = cls.model_json_schema(ref_template="{model}")
    schema.pop("$defs", None)
    schema["refs_are_names"] = True
    return schema


def render(cursor: Cursor, full: bool = False) -> Dict[str, Any]:
    """Render a cursor as the dict ``describe`` returns."""
    node = cursor.node

    if full:
        cls = node.cls if node.kind == "model" else None
        if cls is None or not issubclass(cls, BaseModel):
            raise ValueError(
                f"{cursor.path or '<root>'} is a {node.kind}, which has no schema "
                f"of its own. Ask for one of its members instead."
            )
        return {"path": cursor.path, **shallow_schema(cls)}

    if node.kind == "union" and cursor.view is not None:
        # An instance settles which member is present, so report that member's
        # own fields rather than only the menu -- while still listing the menu,
        # since knowing what one could switch to is the point of asking.
        concrete = cursor.view.concrete_cls(node)
        if concrete is not None:
            payload = _render_concrete(cursor, concrete)
            payload["options"] = _tags(node)
            payload["is"] = type_tag(concrete) or concrete.__name__
            return {"path": cursor.path, **payload}

    if node.kind == "model":
        payload = (
            _render_container(cursor)
            if cursor.adapter is not None
            else _render_class(cursor)
        )
    elif node.kind == "union":
        payload = _render_union(cursor)
    elif node.kind in ("list", "mapping"):
        payload = _render_sequence(cursor)
    else:
        payload = _render_leaf(cursor)

    return {"path": cursor.path, **payload}
