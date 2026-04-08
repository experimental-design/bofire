from typing import Literal

from bofire.data_models.llm.provider import LLMProvider


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
