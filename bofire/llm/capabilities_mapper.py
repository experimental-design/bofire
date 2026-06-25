"""Map LLM capability data models to pydantic-ai capability objects.

Each capability data model (e.g. ``ExperimentAccessCapability``) is mapped to a
pydantic-ai ``AbstractCapability`` via a registered factory function. The data
model carries only serializable config; the factory reconstructs the live
capability — including its tool functions — at mapping time.
"""

from typing import Callable, Optional, Type

import bofire.data_models.llm.api as data_models
from bofire.data_models.llm.capability import ExperimentAccessCapability, LLMCapability


CAPABILITY_MAP: dict[Type[LLMCapability], Callable] = {}


def register(
    data_model_cls: Type[LLMCapability],
    map_fn: Optional[Callable] = None,
):
    """Register a custom capability mapping from data model to factory function.

    Can be used as a decorator or as a direct function call::

        # Decorator form
        @register(MyCapability)
        def map_my_capability(data_model):
            return MyPydanticAICapability(...)

        # Direct call form
        register(MyCapability, map_my_capability)

    Args:
        data_model_cls: The Pydantic data model class.
        map_fn: A callable that takes the data model instance and returns a
            pydantic-ai capability. If not provided, returns a decorator.

    Returns:
        The mapping function (unchanged) when used as a decorator, None otherwise.
    """

    def _register(fn: Callable) -> Callable:
        CAPABILITY_MAP[data_model_cls] = fn

        # Also register with the data model union so Pydantic accepts the type.
        data_models.register_llm_capability(data_model_cls)

        return fn

    if map_fn is not None:
        _register(map_fn)
        return None

    return _register


@register(ExperimentAccessCapability)
def map_experiment_access(data_model: ExperimentAccessCapability):
    from pydantic_ai.capabilities import Toolset

    from bofire.llm.experiment_tools import build_experiment_toolset

    return Toolset(build_experiment_toolset(data_model.max_rows_per_tool_call))


def map(data_model: LLMCapability):
    """Map an LLM capability data model to a pydantic-ai capability instance.

    Args:
        data_model: An LLM capability data model instance.

    Returns:
        A pydantic-ai ``AbstractCapability`` ready to attach to an Agent.

    Raises:
        ValueError: If the capability type is not supported.
    """
    mapper_fn = CAPABILITY_MAP.get(type(data_model))
    if mapper_fn is None:
        supported = ", ".join(c.__name__ for c in CAPABILITY_MAP)
        raise ValueError(
            f"Unsupported LLM capability type: {type(data_model).__name__}. "
            f"Supported: {supported}"
        )
    return mapper_fn(data_model)
