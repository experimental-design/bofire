"""Data model for the LLM-based problem formulator."""

from typing import Any, Dict, Literal, Optional

from pydantic import Field
from pydantic.types import PositiveInt

from bofire.data_models.base import BaseModel
from bofire.data_models.llm.provider import (
    AnthropicFoundryLLMProvider,
    AnthropicLLMProvider,
    AzureOpenAILLMProvider,
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


class FormulatorConfig(BaseModel):
    """Configuration for the LLM-based problem formulator.

    The formulator takes a natural language problem description and produces
    a BoFire Domain by first classifying the problem type, then building the
    domain specification using category-specific prompts.

    Attributes:
        llm: LLM provider configuration.
        model_settings: Optional settings forwarded to pydantic-ai
            (e.g., {"temperature": 0.2, "max_tokens": 4096}).
        max_retries: Maximum retries for domain construction validation.
        system_prompt_override: Optional override for the classification
            system prompt.
    """

    type: Literal["FormulatorConfig"] = "FormulatorConfig"
    llm: AnyLLMProvider
    model_settings: Optional[Dict[str, Any]] = Field(default=None)
    max_retries: PositiveInt = 3
    system_prompt_override: Optional[str] = None
