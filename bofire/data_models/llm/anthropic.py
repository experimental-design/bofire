from typing import Literal, Optional

from pydantic import Field

from bofire.data_models.llm.provider import LLMProvider


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
