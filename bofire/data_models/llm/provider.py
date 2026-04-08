from typing import Any

from bofire.data_models.base import BaseModel


class LLMProvider(BaseModel):
    """Base class for all LLM provider configurations.

    LLM providers are standalone data models that describe how to connect to an
    LLM service. They can be passed to any BoFire component that needs LLM
    capabilities (strategies, agents, etc.). API keys are referenced via
    environment variable names and resolved at runtime.

    Usage-specific settings (temperature, max_tokens, system_prompt) belong on
    the component that uses the provider, not on the provider itself.

    Attributes:
        model: The model identifier (e.g., "claude-sonnet-4-20250514", "gpt-4o").
        api_key_env_var: Name of the environment variable holding the API key.
    """

    type: Any
    model: str
    api_key_env_var: str
