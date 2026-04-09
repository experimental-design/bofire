"""Map LLM provider data models to pydantic-ai Model instances.

Each provider data model (e.g., ``AnthropicLLMProvider``) is mapped to the
corresponding pydantic-ai ``Model`` object. API keys are resolved from
environment variables at mapping time.
"""

import os

from bofire.data_models.llm.anthropic import AnthropicLLMProvider
from bofire.data_models.llm.anthropic_foundry import AnthropicFoundryLLMProvider
from bofire.data_models.llm.openai import OpenAILLMProvider
from bofire.data_models.llm.openai_compatible import OpenAICompatibleLLMProvider
from bofire.data_models.llm.provider import LLMProvider


def _resolve_env_var(env_var_name: str) -> str:
    """Resolve an environment variable, raising if not set."""
    value = os.environ.get(env_var_name)
    if value is None:
        raise EnvironmentError(f"Environment variable '{env_var_name}' is not set.")
    return value


def _map_anthropic(data_model: AnthropicLLMProvider):
    from anthropic import AsyncAnthropic
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

    kwargs = {"api_key": _resolve_env_var(data_model.api_key_env_var)}
    if data_model.base_url is not None:
        kwargs["base_url"] = data_model.base_url

    client = AsyncAnthropic(**kwargs)
    provider = AnthropicProvider(anthropic_client=client)
    return AnthropicModel(data_model.model, provider=provider)


def _map_anthropic_foundry(data_model: AnthropicFoundryLLMProvider):
    from anthropic import AsyncAnthropicFoundry
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

    client = AsyncAnthropicFoundry(
        api_key=_resolve_env_var(data_model.api_key_env_var),
        resource=_resolve_env_var(data_model.resource_env_var),
    )
    provider = AnthropicProvider(anthropic_client=client)
    return AnthropicModel(data_model.model, provider=provider)


def _map_openai(data_model: OpenAILLMProvider):
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


def _map_openai_compatible(data_model: OpenAICompatibleLLMProvider):
    from openai import AsyncOpenAI
    from pydantic_ai.models.openai import OpenAIModel
    from pydantic_ai.providers.openai import OpenAIProvider

    client = AsyncOpenAI(
        api_key=_resolve_env_var(data_model.api_key_env_var),
        base_url=data_model.base_url,
    )
    provider = OpenAIProvider(openai_client=client)
    return OpenAIModel(data_model.model, provider=provider)


_MAP = {
    AnthropicLLMProvider: _map_anthropic,
    AnthropicFoundryLLMProvider: _map_anthropic_foundry,
    OpenAILLMProvider: _map_openai,
    OpenAICompatibleLLMProvider: _map_openai_compatible,
}


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
    mapper_fn = _MAP.get(type(data_model))
    if mapper_fn is None:
        supported = ", ".join(c.__name__ for c in _MAP)
        raise ValueError(
            f"Unsupported LLM provider type: {type(data_model).__name__}. "
            f"Supported: {supported}"
        )
    return mapper_fn(data_model)
