from typing import Annotated, Union

from pydantic import Field

from bofire.data_models.llm.anthropic import AnthropicLLMProvider
from bofire.data_models.llm.anthropic_foundry import AnthropicFoundryLLMProvider
from bofire.data_models.llm.openai import OpenAILLMProvider
from bofire.data_models.llm.openai_compatible import OpenAICompatibleLLMProvider
from bofire.data_models.llm.provider import LLMProvider


AnyLLMProvider = Annotated[
    Union[
        AnthropicLLMProvider,
        AnthropicFoundryLLMProvider,
        OpenAILLMProvider,
        OpenAICompatibleLLMProvider,
    ],
    Field(discriminator="type"),
]

__all__ = [
    "AnyLLMProvider",
    "LLMProvider",
    "AnthropicLLMProvider",
    "AnthropicFoundryLLMProvider",
    "OpenAILLMProvider",
    "OpenAICompatibleLLMProvider",
]
