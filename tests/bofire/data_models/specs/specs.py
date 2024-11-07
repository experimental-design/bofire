import random
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Type


class Spec:
    """Spec <spec> for class <cls>."""

    def __init__(self, cls: Type, spec: Callable[[], dict]):
        self.cls = cls
        self.spec = spec

    def obj(self, **kwargs) -> Any:
        """Create and return an instance of <cls>."""
        return self.cls(**{**self.spec(), **kwargs})

    def typed_spec(self) -> dict:
        """Return the <spec>, extended by {'type': <cls>.__name__}."""
        return {
            **self.spec(),
            "type": self.cls.__name__,
        }

    def __str__(self):
        return f"{self.cls.__name__}: {self.spec()}"

    def __repr__(self):
        return str(self)


class InvalidSpec(Spec):
    def __init__(
        self,
        cls: Type,
        spec: Callable[[], dict],
        error: Exception,
        message: Optional[str] = None,
    ):
        self.cls = cls
        self.spec = spec
        self.error = (error,)
        self.message = message


class Invalidator(ABC):
    """Invalidator rule to invalidate a given spec."""

    @abstractmethod
    def invalidate(self, spec: Spec) -> List[InvalidSpec]:
        """Return a list of invalidated specs.

        If this invalidator is not applicable to the specified
        spec, an empty list is returned.
        """


class Overwrite(Invalidator):
    """Overwrite properties if the key is contained in the spec."""

    def __init__(self, key: str, overwrites: List[dict]):
        self.key = key
        self.overwrites = overwrites

    def invalidate(self, spec: Spec) -> List[InvalidSpec]:
        data: dict = spec.spec()
        if self.key not in data:
            return []
        return [
            InvalidSpec(
                spec.cls,
                lambda data=data, overwrite=overwrite: {**data, **overwrite},
            )
            for overwrite in self.overwrites
        ]


class Specs:
    """Collection of valid and invalid specs.

    In the init, only <invalidators> must be provided.
    Valid specs are added via the <add_valid> method.
    Invalid specs can auomatically be added as part of this method.
    """

    def __init__(self, invalidators: List[Invalidator]):
        self.invalidators = invalidators
        self.valids: List[Spec] = []
        self.invalids: List[InvalidSpec] = []

    def _get_spec(self, specs: List[Spec], cls: Type = None, exact: bool = True):
        if cls is not None:
            if exact:
                specs = [s for s in specs if s.cls == cls]
            else:
                specs = [s for s in specs if issubclass(s.cls, cls)]
        if len(specs) == 0 and cls is None:
            raise TypeError("no spec found")
        if len(specs) == 0:
            raise TypeError(f"no spec of type {cls.__name__} found")
        return random.choice(specs)

    def valid(self, cls: Type = None, exact: bool = True) -> Spec:
        """Return a valid spec.

        If <cls> is provided, the list of all valid specs is filtered by it.
        If no spec (with the specified class) exists, a TypeError is raised.
        If more than one spec exist, a random one is returned.
        """
        return self._get_spec(self.valids, cls, exact)

    def invalid(self, cls: Type = None, exact: bool = True) -> Spec:
        """Return an invalid spec.

        If <cls> is provided, the list of all invalid specs is filtered by it.
        If no spec (with the specified class) exists, a TypeError is raised.
        If more than one spec exist, a random one is returned.
        """
        return self._get_spec(self.invalids, cls, exact)

    def add_valid(
        self,
        cls: Type,
        spec: Callable[[], dict],
        add_invalids: bool = False,
    ) -> Spec:
        """Add a new valid spec to the list.

        If <add_invalids> is True (default), invalid specs are generated using the
        rules provided in <invalidators>.
        """
        spec_ = Spec(cls, spec)
        self.valids.append(spec_)
        if add_invalids:
            for invalidator in self.invalidators:
                self.invalids += invalidator.invalidate(spec_)
        return spec_

    def add_invalid(
        self,
        cls: Type,
        spec: Callable[[], dict],
        error: Exception,
        message: Optional[str] = None,
    ) -> Spec:
        """Add a new invalid spec to the list."""
        spec_ = InvalidSpec(cls, spec, error, message)
        self.invalids.append(spec_)
        return spec_
