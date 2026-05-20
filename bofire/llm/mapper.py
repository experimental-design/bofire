"""Map LLM provider data models to pydantic-ai Model instances.

Each provider data model (e.g., ``AnthropicLLMProvider``) is mapped to the
corresponding pydantic-ai ``Model`` object via a registered factory function.
API keys are resolved from environment variables at mapping time.
"""

import os
from typing import Callable, Optional, Type

import bofire.data_models.llm.api as data_models
from bofire.data_models.llm.provider import (
    AnthropicFoundryLLMProvider,
    AnthropicLLMProvider,
    AzureOpenAILLMProvider,
    LLMProvider,
    OpenAICompatibleLLMProvider,
    OpenAILLMProvider,
)


LLM_MAP: dict[Type[LLMProvider], Callable] = {}


def register(
    data_model_cls: Type[LLMProvider],
    map_fn: Optional[Callable] = None,
):
    """Register a custom LLM provider mapping from data model to factory function.

    Can be used as a decorator or as a direct function call::

        # Decorator form
        @register(MyLLMProvider)
        def map_my_provider(data_model):
            return MyPydanticAIModel(...)

        # Direct call form
        register(MyLLMProvider, map_my_provider)

    Args:
        data_model_cls: The Pydantic data model class.
        map_fn: A callable that takes the data model instance and returns a
            pydantic-ai ``Model``. If not provided, returns a decorator.

    Returns:
        The mapping function (unchanged) when used as a decorator, None otherwise.
    """

    def _register(fn: Callable) -> Callable:
        LLM_MAP[data_model_cls] = fn

        # Also register with the data model union so Pydantic accepts the type
        data_models.register_llm_provider(data_model_cls)

        return fn

    if map_fn is not None:
        _register(map_fn)
        return None

    return _register


def _resolve_env_var(env_var_name: str) -> str:
    """Resolve an environment variable, raising if not set."""
    value = os.environ.get(env_var_name)
    if value is None:
        raise EnvironmentError(f"Environment variable '{env_var_name}' is not set.")
    return value


@register(AnthropicLLMProvider)
def map_anthropic(data_model: AnthropicLLMProvider):
    from anthropic import AsyncAnthropic
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

    kwargs = {"api_key": _resolve_env_var(data_model.api_key_env_var)}
    if data_model.base_url is not None:
        kwargs["base_url"] = data_model.base_url

    client = AsyncAnthropic(**kwargs)
    provider = AnthropicProvider(anthropic_client=client)
    return AnthropicModel(data_model.model, provider=provider)


@register(AnthropicFoundryLLMProvider)
def map_anthropic_foundry(data_model: AnthropicFoundryLLMProvider):
    from anthropic import AsyncAnthropicFoundry
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

    client = AsyncAnthropicFoundry(
        api_key=_resolve_env_var(data_model.api_key_env_var),
        resource=_resolve_env_var(data_model.resource_env_var),
    )
    provider = AnthropicProvider(anthropic_client=client)
    return AnthropicModel(data_model.model, provider=provider)


@register(OpenAILLMProvider)
def map_openai(data_model: OpenAILLMProvider):
    from openai import AsyncOpenAI
    from pydantic_ai.models.openai import OpenAIModel
    from pydantic_ai.providers.openai import OpenAIProvider

    kwargs = {"api_key": _resolve_env_var(data_model.api_key_env_var)}
    if data_model.base_url is not None:
        kwargs["base_url"] = data_model.base_url
    if data_model.organization is not None:
        kwargs["organization"] = data_model.organization

    client = AsyncOpenAI(**kwargs)
    provider = OpenAIProvider(openai_client=client)
    return OpenAIModel(data_model.model, provider=provider)


@register(AzureOpenAILLMProvider)
def map_azure_openai(data_model: AzureOpenAILLMProvider):
    from openai import AsyncAzureOpenAI
    from pydantic_ai.models.openai import OpenAIModel
    from pydantic_ai.providers.openai import OpenAIProvider

    client = AsyncAzureOpenAI(
        api_key=_resolve_env_var(data_model.api_key_env_var),
        azure_endpoint=_resolve_env_var(data_model.azure_endpoint_env_var),
        api_version=data_model.api_version,
    )
    provider = OpenAIProvider(openai_client=client)
    return OpenAIModel(data_model.model, provider=provider)


@register(OpenAICompatibleLLMProvider)
def map_openai_compatible(data_model: OpenAICompatibleLLMProvider):
    from openai import AsyncOpenAI
    from pydantic_ai.models.openai import OpenAIModel
    from pydantic_ai.providers.openai import OpenAIProvider

    client = AsyncOpenAI(
        api_key=_resolve_env_var(data_model.api_key_env_var),
        base_url=data_model.base_url,
    )
    provider = OpenAIProvider(openai_client=client)
    return OpenAIModel(data_model.model, provider=provider)


def map(data_model: LLMProvider):
    """Map an LLM provider data model to a pydantic-ai Model instance.

    Args:
        data_model: An LLM provider data model instance.

    Returns:
        A pydantic-ai Model ready for use with pydantic-ai Agent.

    Raises:
        EnvironmentError: If required environment variables are not set.
        ValueError: If the provider type is not supported.
    """
    mapper_fn = LLM_MAP.get(type(data_model))
    if mapper_fn is None:
        supported = ", ".join(c.__name__ for c in LLM_MAP)
        raise ValueError(
            f"Unsupported LLM provider type: {type(data_model).__name__}. "
            f"Supported: {supported}"
        )
    return mapper_fn(data_model)
