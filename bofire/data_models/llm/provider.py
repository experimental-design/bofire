from typing import Any, Literal, Optional

from pydantic import Field

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


class AnthropicLLMProvider(LLMProvider):
    """LLM provider for the direct Anthropic API.

    Attributes:
        model: Anthropic model identifier.
        api_key_env_var: Environment variable name for the Anthropic API key.
        base_url: Optional custom base URL for the API.
    """

    type: Literal["AnthropicLLMProvider"] = "AnthropicLLMProvider"
    model: str = "claude-sonnet-4-20250514"
    api_key_env_var: str = "ANTHROPIC_API_KEY"
    base_url: Optional[str] = Field(
        default=None,
        description="Custom base URL for the Anthropic API.",
    )


class AnthropicFoundryLLMProvider(LLMProvider):
    """LLM provider for Anthropic models hosted on Azure AI Foundry.

    Attributes:
        model: Anthropic model identifier on Azure.
        api_key_env_var: Environment variable name for the Foundry API key.
        resource_env_var: Environment variable name for the Azure resource name.
    """

    type: Literal["AnthropicFoundryLLMProvider"] = "AnthropicFoundryLLMProvider"
    model: str = "claude-sonnet-4-20250514"
    api_key_env_var: str = "ANTHROPIC_FOUNDRY_API_KEY"
    resource_env_var: str = "ANTHROPIC_FOUNDRY_RESOURCE"


class OpenAILLMProvider(LLMProvider):
    """LLM provider for the OpenAI API (including Azure OpenAI via base_url).

    Attributes:
        model: OpenAI model identifier.
        api_key_env_var: Environment variable name for the OpenAI API key.
        base_url: Optional custom base URL (e.g., for Azure OpenAI).
        organization: Optional OpenAI organization ID.
    """

    type: Literal["OpenAILLMProvider"] = "OpenAILLMProvider"
    model: str = "gpt-4o"
    api_key_env_var: str = "OPENAI_API_KEY"
    base_url: Optional[str] = Field(
        default=None,
        description="Custom base URL for the API (e.g., Azure OpenAI endpoint).",
    )
    organization: Optional[str] = Field(
        default=None,
        description="OpenAI organization ID.",
    )


class AzureOpenAILLMProvider(LLMProvider):
    """LLM provider for Azure OpenAI Service.

    Attributes:
        model: The Azure deployment name.
        api_key_env_var: Environment variable name for the Azure OpenAI API key.
        azure_endpoint_env_var: Environment variable name for the Azure endpoint URL.
        api_version: Azure OpenAI API version.
    """

    type: Literal["AzureOpenAILLMProvider"] = "AzureOpenAILLMProvider"
    model: str = "gpt-4o"
    api_key_env_var: str = "AZURE_OPENAI_API_KEY"
    azure_endpoint_env_var: str = "AZURE_OPENAI_ENDPOINT"
    api_version: str = "2024-12-01-preview"


class OpenAICompatibleLLMProvider(LLMProvider):
    """LLM provider for any OpenAI-compatible API (vLLM, Ollama, etc.).

    Attributes:
        model: Model identifier at the endpoint.
        api_key_env_var: Environment variable name for the API key.
        base_url: The endpoint URL (required).
    """

    type: Literal["OpenAICompatibleLLMProvider"] = "OpenAICompatibleLLMProvider"
    model: str
    api_key_env_var: str = "OPENAI_API_KEY"
    base_url: str
