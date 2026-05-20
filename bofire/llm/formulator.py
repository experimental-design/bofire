"""LLM-based problem formulator: natural language → BoFire Domain.

This module implements the two-step formulation pipeline:
1. CLASSIFY: Determine the problem category (cheap LLM call, enum output)
2. BUILD: Generate a DomainSpec using category-specific prompts (structured output)

The DomainSpec is then converted to a BoFire Domain via a deterministic converter,
with a retry loop that feeds validation errors back to the LLM.
"""

import asyncio
import warnings
from typing import Optional

from pydantic_ai import Agent, ModelRetry

import bofire.llm.mapper as llm_mapper
from bofire.data_models.llm.formulator import FormulatorConfig
from bofire.llm.formulator_converter import convert_domain_spec
from bofire.llm.formulator_prompts import CLASSIFY_SYSTEM_PROMPT, get_build_prompt
from bofire.llm.formulator_schemas import (
    ClassificationResult,
    DomainSpec,
    FormulationResult,
)


def formulate(config: FormulatorConfig, description: str) -> FormulationResult:
    """Formulate a natural language problem as a BoFire Domain.

    This is the main entry point. It:
    1. Classifies the problem into a category
    2. If experimental: builds a DomainSpec and converts to Domain
    3. If not_experimental: returns a warning with no Domain

    Args:
        config: Formulator configuration (LLM provider, settings, retries).
        description: Natural language problem description.

    Returns:
        FormulationResult with the Domain (or None if not_experimental),
        classification, and reasoning.

    Raises:
        ValueError: If the Domain cannot be constructed after all retries.
    """
    return asyncio.run(_formulate_async(config, description))


async def _formulate_async(
    config: FormulatorConfig, description: str
) -> FormulationResult:
    """Async implementation of the formulation pipeline."""
    model = llm_mapper.map(config.llm)

    # Step 1: Classify
    classification = await _classify(model, config, description)

    # Step 2: Handle not_experimental
    if classification.category == "not_experimental":
        warnings.warn(
            "This problem was classified as 'not_experimental' (mathematical "
            "programming / combinatorial optimization). BoFire is designed for "
            "experimental design and Bayesian optimization of physical experiments. "
            "Consider using a mathematical programming solver (scipy.optimize.linprog, "
            "PuLP, or Gurobi) instead.\n\n"
            f"Reasoning: {classification.reasoning}",
            UserWarning,
            stacklevel=4,
        )
        return FormulationResult(
            domain=None,
            classification=classification.category,
            reasoning=classification.reasoning,
            domain_spec=None,
        )

    # Step 3: Build DomainSpec with category-specific prompt
    domain_spec = await _build_domain_spec(model, config, description, classification)

    # Step 4: Convert and validate with retry loop
    domain = await _convert_with_retries(
        model, config, description, classification, domain_spec
    )

    return FormulationResult(
        domain=domain,
        classification=classification.category,
        reasoning=classification.reasoning,
        domain_spec=domain_spec,
    )


async def _classify(model, config: FormulatorConfig, description: str):
    """Step 1: Classify the problem into a category."""
    system_prompt = config.system_prompt_override or CLASSIFY_SYSTEM_PROMPT

    agent = Agent(
        model,
        system_prompt=system_prompt,
        output_type=ClassificationResult,
        name="Formulator-Classify",
    )

    result = await agent.run(
        f"Classify this optimization/experimental design problem:\n\n{description}",
        model_settings=config.model_settings,
    )
    return result.output


async def _build_domain_spec(
    model,
    config: FormulatorConfig,
    description: str,
    classification: ClassificationResult,
) -> DomainSpec:
    """Step 2: Build the DomainSpec using a category-specific prompt."""
    build_prompt = get_build_prompt(classification.category)

    agent = Agent(
        model,
        system_prompt=build_prompt,
        output_type=DomainSpec,
        name="Formulator-Build",
    )

    user_message = (
        f"Problem classification: {classification.category}\n"
        f"Classification reasoning: {classification.reasoning}\n\n"
        f"Problem description:\n{description}\n\n"
        f"Please produce the structured domain specification."
    )

    result = await agent.run(
        user_message,
        model_settings=config.model_settings,
    )
    return result.output


async def _convert_with_retries(
    model,
    config: FormulatorConfig,
    description: str,
    classification: ClassificationResult,
    domain_spec: DomainSpec,
):
    """Step 3: Convert DomainSpec to Domain, retrying on validation errors."""
    last_error: Optional[str] = None

    # First attempt: direct conversion
    try:
        return convert_domain_spec(domain_spec)
    except Exception as e:
        last_error = str(e)

    # Retry loop: ask LLM to fix the spec based on the error
    build_prompt = get_build_prompt(classification.category)

    agent = Agent(
        model,
        system_prompt=build_prompt,
        output_type=DomainSpec,
        output_retries=config.max_retries,
        name="Formulator-Retry",
    )

    @agent.output_validator
    async def validate_domain_spec(ctx, spec: DomainSpec):
        try:
            convert_domain_spec(spec)
        except Exception as e:
            raise ModelRetry(
                f"Domain construction failed with error:\n{e}\n\n"
                f"Please fix the specification to resolve this error."
            ) from e
        return spec

    retry_message = (
        f"Problem description:\n{description}\n\n"
        f"The previous domain specification failed validation with this error:\n"
        f"{last_error}\n\n"
        f"Please produce a corrected domain specification."
    )

    result = await agent.run(
        retry_message,
        model_settings=config.model_settings,
    )

    # The validator passed, so conversion will succeed
    return convert_domain_spec(result.output)
