from typing import Any, Literal, Optional

from pydantic import Field

from bofire.data_models.base import BaseModel


class LLMProvider(BaseModel):
    """Connection settings for an LLM service, which can be passed to any BoFire
    component that uses an LLM."""

    type: Any
    model: str = Field(
        description="Identifier of the model at the service, e.g. "
        "'claude-sonnet-4-20250514' or 'gpt-4o'.",
    )
    api_key_env_var: str = Field(
        description="Name of the environment variable holding the API key.",
    )


class AnthropicLLMProvider(LLMProvider):
    """LLM provider for the Anthropic API."""

    type: Literal["AnthropicLLMProvider"] = "AnthropicLLMProvider"
    model: str = Field(
        default="claude-sonnet-4-20250514", description="Anthropic model identifier."
    )
    api_key_env_var: str = Field(
        default="ANTHROPIC_API_KEY",
        description="Name of the environment variable holding the Anthropic API key.",
    )
    base_url: Optional[str] = Field(
        default=None,
        description="Custom base URL for the Anthropic API.",
    )


class AnthropicFoundryLLMProvider(LLMProvider):
    """LLM provider for Anthropic models hosted on Azure AI Foundry."""

    type: Literal["AnthropicFoundryLLMProvider"] = "AnthropicFoundryLLMProvider"
    model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Anthropic model identifier on Azure.",
    )
    api_key_env_var: str = Field(
        default="ANTHROPIC_FOUNDRY_API_KEY",
        description="Name of the environment variable holding the Foundry API key.",
    )
    resource_env_var: str = Field(
        default="ANTHROPIC_FOUNDRY_RESOURCE",
        description="Name of the environment variable holding the Azure resource "
        "name.",
    )


class OpenAILLMProvider(LLMProvider):
    """LLM provider for the OpenAI API, including Azure OpenAI via `base_url`."""

    type: Literal["OpenAILLMProvider"] = "OpenAILLMProvider"
    model: str = Field(default="gpt-4o", description="OpenAI model identifier.")
    api_key_env_var: str = Field(
        default="OPENAI_API_KEY",
        description="Name of the environment variable holding the OpenAI API key.",
    )
    base_url: Optional[str] = Field(
        default=None,
        description="Custom base URL for the API (e.g., Azure OpenAI endpoint).",
    )
    organization: Optional[str] = Field(
        default=None,
        description="OpenAI organization ID.",
    )


class OpenAICompatibleLLMProvider(LLMProvider):
    """LLM provider for any OpenAI-compatible API, such as vLLM or Ollama."""

    type: Literal["OpenAICompatibleLLMProvider"] = "OpenAICompatibleLLMProvider"
    api_key_env_var: str = Field(
        default="OPENAI_API_KEY",
        description="Name of the environment variable holding the API key.",
    )
    base_url: str = Field(description="URL of the endpoint.")
