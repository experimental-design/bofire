from typing import Literal, Optional

from pydantic import Field

from bofire.data_models.llm.provider import LLMProvider


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
