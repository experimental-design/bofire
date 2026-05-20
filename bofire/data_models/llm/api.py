from bofire.data_models.llm._register import register_llm_provider  # noqa: F401
from bofire.data_models.llm.formulator import FormulatorConfig  # noqa: F401
from bofire.data_models.llm.provider import (
    AnthropicFoundryLLMProvider,
    AnthropicLLMProvider,
    AzureOpenAILLMProvider,
    LLMProvider,  # noqa: F401
    OpenAICompatibleLLMProvider,
    OpenAILLMProvider,
)
from bofire.data_models.unions import tagged_union


AnyLLMProvider = tagged_union(
    AnthropicLLMProvider,
    AnthropicFoundryLLMProvider,
    AzureOpenAILLMProvider,
    OpenAILLMProvider,
    OpenAICompatibleLLMProvider,
)
