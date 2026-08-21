"""A uniform view over an instance, which may be a model object or a raw dict.

``describe(path, instance=...)`` accepts either a pydantic object or a
serialized spec. It must not validate a dict into a model first: agents describe
half-built specs, and some containers cannot self-identify anyway --
``BotorchSurrogates`` declares no ``type`` field at all, so a bare dict for it
carries no discriminator.

:class:`View` therefore wraps whatever it is given and answers the few questions
the resolver asks, tolerating absence rather than raising: a field missing from a
partial dict yields a view with ``present`` false.
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

from pydantic import BaseModel

from bofire.data_models.introspection.annotations import TypeNode, type_tag
from bofire.data_models.introspection.containers import ContainerAdapter


__all__ = ["View"]


@dataclass(frozen=True)
class View:
    """An instance, or the absence of one, at some location in a spec."""

    obj: Any
    present: bool = True
    #: The container of ``obj`` and the accessor within it, for the write path.
    parent: Any = None
    accessor: Optional[Union[str, int]] = None

    @property
    def is_missing(self) -> bool:
        return not self.present

    def field(self, name: str) -> "View":
        """Descend into a named field."""
        if self.is_missing:
            return self
        if isinstance(self.obj, dict):
            if name not in self.obj:
                return View(None, present=False, parent=self.obj, accessor=name)
            return View(self.obj[name], parent=self.obj, accessor=name)
        return View(
            getattr(self.obj, name, None),
            present=hasattr(self.obj, name),
            parent=self.obj,
            accessor=name,
        )

    def items(self) -> List[Any]:
        """The raw elements, when this view wraps a list."""
        if self.is_missing or self.obj is None:
            return []
        if isinstance(self.obj, (list, tuple)):
            return list(self.obj)
        # A container model reached directly, e.g. via ``full_schema``.
        return []

    def element(self, index: int) -> "View":
        """Descend into a list position."""
        elements = self.items()
        if index < -len(elements) or index >= len(elements):
            return View(None, present=False, parent=self.obj, accessor=index)
        return View(elements[index], parent=self.obj, accessor=index)

    def keys(self, adapter: ContainerAdapter) -> List[str]:
        """The element keys, for a keyed container's list."""
        keys = [adapter.key_of(element) for element in self.items()]
        return [key for key in keys if key is not None]

    def index_of_key(self, adapter: ContainerAdapter, key: str) -> Optional[int]:
        """The position of the element carrying *key*, if any."""
        for index, element in enumerate(self.items()):
            if adapter.key_of(element) == key:
                return index
        return None

    def concrete_cls(self, node: TypeNode) -> Optional[type]:
        """Which member of a union node this instance actually is.

        Returns ``None`` when the instance is absent or carries no usable
        discriminator, which the resolver reports rather than guessing.
        """
        if self.is_missing or self.obj is None:
            return None
        if isinstance(self.obj, BaseModel):
            actual = type(self.obj)
            for option in node.options:
                if actual is option or issubclass(actual, option):
                    return actual
            return actual
        if isinstance(self.obj, dict):
            tag = self.obj.get(node.discriminator or "type")
            if not isinstance(tag, str):
                return None
            for option in node.options:
                if type_tag(option) == tag or option.__name__ == tag:
                    return option
        return None

    def json_value(self) -> Any:
        """A JSON-safe rendering of a leaf value."""
        obj = self.obj
        if isinstance(obj, BaseModel):
            return obj.model_dump()
        if isinstance(obj, (list, tuple)):
            return [View(item).json_value() for item in obj]
        if isinstance(obj, dict):
            return {k: View(v).json_value() for k, v in obj.items()}
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        return str(obj)

    def target(self) -> Tuple[Any, Optional[Union[str, int]]]:
        """The ``(owner, accessor)`` pair a write would use.

        Assigning through it re-validates, because BoFire data models set
        ``validate_assignment=True``.
        """
        return self.parent, self.accessor
