from bofire.data_models.llm._register import (  # noqa: F401
    register_llm_capability,
    register_llm_provider,
)
from bofire.data_models.llm.capability import (
    ExperimentAccessCapability,
    LLMCapability,  # noqa: F401
)
from bofire.data_models.llm.provider import (
    AnthropicFoundryLLMProvider,
    AnthropicLLMProvider,
    LLMProvider,  # noqa: F401
    OpenAICompatibleLLMProvider,
    OpenAILLMProvider,
)
from bofire.data_models.unions import tagged_union


AnyLLMProvider = tagged_union(
    AnthropicLLMProvider,
    AnthropicFoundryLLMProvider,
    OpenAILLMProvider,
    OpenAICompatibleLLMProvider,
)

AnyLLMCapability = tagged_union(
    ExperimentAccessCapability,
)
