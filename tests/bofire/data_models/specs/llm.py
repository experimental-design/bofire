from bofire.data_models.llm.anthropic import AnthropicLLMProvider
from bofire.data_models.llm.anthropic_foundry import AnthropicFoundryLLMProvider
from bofire.data_models.llm.openai import OpenAILLMProvider
from bofire.data_models.llm.openai_compatible import OpenAICompatibleLLMProvider
from tests.bofire.data_models.specs.specs import Specs


specs = Specs([])

specs.add_valid(
    AnthropicLLMProvider,
    lambda: {
        "model": "claude-sonnet-4-20250514",
        "api_key_env_var": "ANTHROPIC_API_KEY",
        "base_url": None,
    },
)

specs.add_valid(
    AnthropicFoundryLLMProvider,
    lambda: {
        "model": "claude-sonnet-4-20250514",
        "api_key_env_var": "ANTHROPIC_FOUNDRY_API_KEY",
        "resource_env_var": "ANTHROPIC_FOUNDRY_RESOURCE",
    },
)

specs.add_valid(
    OpenAILLMProvider,
    lambda: {
        "model": "gpt-4o",
        "api_key_env_var": "OPENAI_API_KEY",
        "base_url": None,
        "organization": None,
    },
)

specs.add_valid(
    OpenAICompatibleLLMProvider,
    lambda: {
        "model": "my-model",
        "api_key_env_var": "CUSTOM_API_KEY",
        "base_url": "http://localhost:8000/v1",
    },
)
