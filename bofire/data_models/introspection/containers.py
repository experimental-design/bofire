"""How container models address their elements in a serialized spec.

``Inputs``/``Outputs`` wrap a list of features addressed by feature key, and
``BotorchSurrogates`` wraps a list of surrogates addressed by the output each
one models. ``Constraints`` wraps a list too, but constraints have no key, so
its elements are addressed by position.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, List, Optional, Tuple, Union


__all__ = ["Container", "container_for"]


@dataclass(frozen=True)
class Container:
    """Which field of a container model holds its elements, and how they are keyed.

    ``key_path`` leads from a serialized element to its key; it is empty for
    containers whose elements are addressed by position only.
    """

    list_field: str
    key_path: Tuple[Union[str, int], ...] = ()

    @property
    def keyed(self) -> bool:
        return bool(self.key_path)

    def key_of(self, element: Any) -> Optional[str]:
        """The key of a serialized element, or ``None`` if it has none."""
        current = element
        for step in self.key_path:
            if isinstance(current, dict):
                current = current.get(step)
            elif isinstance(current, list) and isinstance(step, int):
                current = (
                    current[step] if -len(current) <= step < len(current) else None
                )
            else:
                return None
        return current if self.keyed and isinstance(current, str) else None


#: Elements that carry their own ``key`` field, as every feature does.
KEY_FIELD = Container(list_field="", key_path=("key",))


@lru_cache(maxsize=1)
def _containers() -> List[Tuple[type, Container]]:
    from bofire.data_models.domain.constraints import Constraints
    from bofire.data_models.domain.features import _BaseFeatures
    from bofire.data_models.surrogates.botorch_surrogates import BotorchSurrogates

    return [
        (
            BotorchSurrogates,
            Container("surrogates", ("outputs", "features", 0, "key")),
        ),
        (_BaseFeatures, Container("features", ("key",))),
        (Constraints, Container("constraints")),
    ]


def container_for(cls: Any) -> Optional[Container]:
    """The container description for *cls*, matched by subclass, or ``None``."""
    if not isinstance(cls, type):
        return None
    for container_cls, container in _containers():
        if issubclass(cls, container_cls):
            return container
    return None
