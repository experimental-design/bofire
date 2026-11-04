from bofire.data_models.llm._register import register_llm_provider  # noqa: F401
from bofire.data_models.llm.provider import (
    AnthropicFoundryLLMProvider,
    AnthropicLLMProvider,
    LLMProvider,  # noqa: F401
    OpenAICompatibleLLMProvider,
    OpenAILLMProvider,
)
from bofire.data_models.unions import tagged_union


_LLM_PROVIDER_TYPES: list[type[LLMProvider]] = [
    AnthropicLLMProvider,
    AnthropicFoundryLLMProvider,
    OpenAILLMProvider,
    OpenAICompatibleLLMProvider,
]

AnyLLMProvider = tagged_union(*_LLM_PROVIDER_TYPES)
