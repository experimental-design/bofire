from typing import Any, Dict, Literal, Optional, Type

from pydantic import Field, PositiveInt, model_validator

from bofire.data_models.constraints.api import (
    Constraint,
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
)
from bofire.data_models.features.api import Feature
from bofire.data_models.llm.api import AnyLLMProvider
from bofire.data_models.objectives.api import MaximizeObjective, MinimizeObjective
from bofire.data_models.strategies.strategy import Strategy


_DEFAULT_SYSTEM_PROMPT = """\
You are an expert experimental design assistant for Bayesian optimization.

You will receive a description of an optimization problem including:
- Objectives (what to maximize/minimize)
- Constraints (mathematical relationships candidates MUST satisfy)
- Problem context and domain knowledge

The output schema already encodes the input features with their types,
bounds, and allowed values. Use the field descriptions to understand each feature.

Consider:
- Domain knowledge from the context descriptions
- Feature bounds and constraints (candidates MUST be feasible)
- Diversity of proposed candidates (explore the space)
- If previous experiments are provided, use them to inform proposals
For each candidate, briefly explain your reasoning.
"""


class LLMStrategy(Strategy):
    """Lets a large language model propose the candidates.

    Instead of fitting a surrogate and optimizing an acquisition function, the model
    reads a description of the problem (inputs with their bounds and values,
    constraints, the objective, and the `context` of the domain and its features)
    together with previous experiments, and proposes candidates with a short
    reasoning for each. Useful for cold starts, for categorical or molecular inputs
    where domain knowledge helps, and when written context matters. There is no
    uncertainty model behind the proposals, so they are informed guesses rather
    than optima.

    Proposed candidates are checked against the bounds and constraints of the
    domain; violations are sent back to the model, which gets `output_retries`
    attempts to correct them. Supports a single output with a maximize or minimize
    objective, all input types, and linear and NChooseK constraints. The returned
    candidates carry the model's reasoning in an extra column.

    Examples:
        >>> LLMStrategy(
        ...     domain=domain,
        ...     llm=AnthropicLLMProvider(model="claude-sonnet-5-5"),
        ...     model_settings={"thinking": "high"},
        ... )
    """

    type: Literal["LLMStrategy"] = "LLMStrategy"

    llm: AnyLLMProvider = Field(
        description="The LLM service and model that propose the candidates.",
    )
    model_settings: Dict[str, Any] = Field(
        default={},
        description="Settings passed on to the model, as understood by pydantic-ai "
        "and the provider, e.g. 'temperature', 'max_tokens', 'seed', 'timeout', or "
        "'thinking' ('low', 'medium' or 'high') for extended reasoning on hard "
        "problems, at a considerably higher cost and latency.",
    )
    output_retries: PositiveInt = Field(
        default=3,
        description="How often the model may correct candidates that violate bounds "
        "or constraints; each retry shows it the invalid candidates and the error.",
    )
    n_recent_experiments: Optional[PositiveInt] = Field(
        default=None,
        description="If provided, only this many of the most recent experiments are "
        "shown to the model, to bound the prompt on long campaigns.",
    )
    n_top_experiments: Optional[PositiveInt] = Field(
        default=None,
        description="If provided, this many experiments with the best objective "
        "values are shown, in addition to the recent ones if `n_recent_experiments` "
        "is also given. If neither is provided, all experiments are shown.",
    )
    system_prompt: str = Field(
        default=_DEFAULT_SYSTEM_PROMPT,
        description="Instructions the model receives as system prompt. The problem "
        "description and the experiments are added to it in any case.",
    )

    @model_validator(mode="after")
    def validate_single_objective(self):
        """Validate that the domain has exactly one output with a supported objective."""
        outputs_with_obj = self.domain.outputs.get_by_objective(
            includes=[MaximizeObjective, MinimizeObjective],
        )
        if len(outputs_with_obj) != 1:
            raise ValueError(
                f"LLMStrategy requires exactly one output with a Maximize or "
                f"Minimize objective, got {len(outputs_with_obj)}."
            )
        return self

    def is_constraint_implemented(self, my_type: Type[Constraint]) -> bool:
        return my_type in [
            LinearEqualityConstraint,
            LinearInequalityConstraint,
            NChooseKConstraint,
        ]

    @classmethod
    def is_feature_implemented(cls, my_type: Type[Feature]) -> bool:
        return True
