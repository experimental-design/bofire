"""Intermediate schemas for the LLM formulator.

These Pydantic models define the structured output that the LLM produces.
They are intentionally simpler than BoFire's full Domain type system so that
LLMs can reliably fill them. A deterministic converter (formulator_converter.py)
then maps these to proper BoFire Domain objects.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from bofire.data_models.domain.api import Domain


# --- Classification step output ---

ProblemCategory = Literal[
    "initial_design",
    "single_objective",
    "multi_objective",
    "space_filling",
    "not_experimental",
]


class ClassificationResult(BaseModel):
    """Result of the problem classification step."""

    category: ProblemCategory = Field(
        description=(
            "The problem category. Use 'initial_design' when the user wants to "
            "design experiments to explore a space or fit a model. Use "
            "'single_objective' when optimizing exactly one measurable outcome. "
            "Use 'multi_objective' when there are multiple competing objectives. "
            "Use 'space_filling' when the goal is uniform coverage without a "
            "specific optimization target. Use 'not_experimental' when the "
            "problem is a mathematical program (LP, MILP, scheduling, assignment) "
            "that does not involve physical experiments."
        )
    )
    reasoning: str = Field(
        description="Brief explanation of why this category was chosen."
    )


# --- Domain specification step output ---


class InputSpec(BaseModel):
    """Specification for a single input feature."""

    name: str = Field(description="Short snake_case identifier for the variable.")
    type: Literal["continuous", "discrete", "categorical"] = Field(
        description="The type of input variable."
    )
    description: Optional[str] = Field(
        default=None, description="What this variable represents."
    )
    unit: Optional[str] = Field(default=None, description="Physical unit if any.")
    # For continuous inputs
    lower_bound: Optional[float] = Field(
        default=None, description="Lower bound (required for continuous)."
    )
    upper_bound: Optional[float] = Field(
        default=None, description="Upper bound (required for continuous)."
    )
    # For discrete inputs
    values: Optional[List[float]] = Field(
        default=None,
        description="Allowed discrete values (required for discrete type).",
    )
    # For categorical inputs
    categories: Optional[List[str]] = Field(
        default=None,
        description="Allowed category names (required for categorical type).",
    )


class OutputSpec(BaseModel):
    """Specification for a single output feature."""

    name: str = Field(description="Short snake_case identifier for the output.")
    objective: Literal["maximize", "minimize", "close_to_target"] = Field(
        description="The optimization direction for this output."
    )
    target_value: Optional[float] = Field(
        default=None,
        description="Target value (required when objective is 'close_to_target').",
    )
    description: Optional[str] = Field(
        default=None, description="What this output measures."
    )


class ConstraintSpec(BaseModel):
    """Specification for a linear constraint."""

    type: Literal["linear_equality", "linear_inequality"] = Field(
        description=(
            "Use 'linear_equality' for sum constraints (e.g., fractions sum to 1). "
            "Use 'linear_inequality' for upper-bound constraints (coefficients * x <= rhs)."
        )
    )
    features: List[str] = Field(
        description="Names of the input features involved in this constraint."
    )
    coefficients: List[float] = Field(
        description="Coefficients for each feature in the linear expression."
    )
    rhs: float = Field(description="Right-hand side value of the constraint.")
    description: Optional[str] = Field(
        default=None, description="What this constraint represents physically."
    )


class DomainSpec(BaseModel):
    """Complete domain specification produced by the LLM.

    This is the structured output the LLM fills in. It gets converted
    to a BoFire Domain by the deterministic converter.
    """

    inputs: List[InputSpec] = Field(
        min_length=1,
        description="The controllable input variables (decision variables / factors).",
    )
    outputs: List[OutputSpec] = Field(
        min_length=1,
        description="The measurable output variables (responses / KPIs).",
    )
    constraints: List[ConstraintSpec] = Field(
        default_factory=list,
        description="Linear constraints between input features (if any).",
    )
    context: Optional[str] = Field(
        default=None,
        description="A brief summary of the problem for documentation.",
    )


# --- Final result returned to the user ---


class FormulationResult(BaseModel):
    """Result of the formulation process."""

    domain: Optional[Domain] = Field(
        default=None,
        description="The constructed BoFire Domain (None if not_experimental).",
    )
    classification: ProblemCategory = Field(
        description="The problem category determined by the classifier."
    )
    reasoning: str = Field(
        description="Explanation of the classification and formulation decisions."
    )
    domain_spec: Optional[DomainSpec] = Field(
        default=None,
        description="The intermediate domain specification (for debugging).",
    )

    model_config = {"arbitrary_types_allowed": True}
