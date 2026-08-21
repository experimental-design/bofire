"""Adapters describing how a container model addresses its elements.

Several data models wrap a single list and are naturally addressed by a key
rather than a position: ``Inputs``/``Outputs`` by feature key, and
``BotorchSurrogates`` by the output key its surrogate models. ``Constraints``
wraps a list too, but a constraint has no key, so it stays index-addressed --
which is also what its own ``__getitem__`` accepts.

An adapter states which field holds the list and, when the elements are keyed,
where inside an element the key lives. The key path is navigated the same way on
a pydantic object and on a serialized dict, so one declaration serves both.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, List, Optional, Tuple, Union


__all__ = ["ContainerAdapter", "adapter_for"]

KeyStep = Union[str, int]


@dataclass(frozen=True)
class ContainerAdapter:
    """How to enumerate and address the elements of a container model."""

    list_field: str
    keyed: bool
    key_path: Tuple[KeyStep, ...] = ()

    @property
    def addressing(self) -> str:
        """``"key"`` or ``"index"``, as reported by ``describe``."""
        return "key" if self.keyed else "index"

    def key_of(self, element: Any) -> Optional[str]:
        """Extract an element's key, from either a model or a serialized dict.

        Returns ``None`` if the container is unkeyed or the element is too
        partial to carry a key -- agents legitimately describe half-built specs.
        """
        if not self.keyed:
            return None
        current = element
        for step in self.key_path:
            current = _navigate(current, step)
            if current is None:
                return None
        return current if isinstance(current, str) else None

    def replace_element(self, owner: Any, index: int, value: Any) -> None:
        """Replace one element by reassigning the whole list.

        Mutating a ``Sequence`` element in place would bypass
        ``validate_assignment``, which is how BoFire data models re-validate on
        write. Used by the write path; ``describe`` never calls it.
        """
        if isinstance(owner, dict):
            owner[self.list_field][index] = value
            return
        items = list(getattr(owner, self.list_field))
        items[index] = value
        setattr(owner, self.list_field, items)


def _navigate(obj: Any, step: KeyStep) -> Any:
    """Follow one step into a model, dict or sequence, tolerating absence."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(step)
    if isinstance(step, int):
        try:
            return obj[step]
        except (IndexError, KeyError, TypeError):
            return None
    return getattr(obj, step, None)


@lru_cache(maxsize=1)
def _adapters() -> List[Tuple[type, ContainerAdapter]]:
    """Container classes paired with their adapter, most specific first.

    Imports are deferred so that importing this module does not pull in the
    whole data model package.
    """
    from bofire.data_models.domain.constraints import Constraints
    from bofire.data_models.domain.features import _BaseFeatures
    from bofire.data_models.surrogates.botorch_surrogates import BotorchSurrogates

    return [
        (
            BotorchSurrogates,
            ContainerAdapter(
                list_field="surrogates",
                keyed=True,
                # A surrogate is addressed by the output it models;
                # ``BotorchSurrogates`` enforces that output keys are unique.
                key_path=("outputs", "features", 0, "key"),
            ),
        ),
        (
            _BaseFeatures,
            ContainerAdapter(list_field="features", keyed=True, key_path=("key",)),
        ),
        (
            Constraints,
            ContainerAdapter(list_field="constraints", keyed=False),
        ),
    ]


def adapter_for(cls: Any) -> Optional[ContainerAdapter]:
    """Return the adapter for a container class, or ``None`` if it is not one.

    Matching is by subclass, so registered custom ``Inputs``/``Outputs``
    subclasses are handled without being listed.
    """
    if not isinstance(cls, type):
        return None
    for container_cls, adapter in _adapters():
        if issubclass(cls, container_cls):
            return adapter
    return None
