from typing import Literal

from bofire.data_models.llm.provider import LLMProvider


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
