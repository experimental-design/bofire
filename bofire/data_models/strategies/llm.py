from typing import Annotated, Any, Dict, Literal, Optional, Type

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


class LLMStrategy(Strategy):
    """Strategy that uses an LLM to propose optimization candidates.

    Uses pydantic-ai structured output with a dynamically generated schema
    that matches the Domain's input features. The LLM receives feature bounds,
    allowed values, objectives, constraints, and context as prompt context.
    Domain validation catches constraint violations, and pydantic-ai retries
    automatically.

    Currently supports single-objective optimization with Maximize or Minimize
    objectives, and linear/NChooseK constraints.

    Attributes:
        llm: LLM provider configuration.
        model_settings: Optional dict forwarded directly to pydantic-ai's
            ``model_settings`` (e.g. ``{"temperature": 0.2, "max_tokens": 4096,
            "thinking": "high"}``). Keys are not validated by BoFire — pydantic-ai
            and the underlying provider SDK are the source of truth.
        output_retries: Number of retries when output validation fails.
        n_recent_experiments: Number of most recent experiments to show the LLM.
        n_top_experiments: Number of top-performing experiments to show the LLM.
        system_prompt: Optional override for the default system prompt.
    """

    type: Literal["LLMStrategy"] = "LLMStrategy"

    llm: AnyLLMProvider
    model_settings: Optional[Dict[str, Any]] = None
    output_retries: PositiveInt = 3
    n_recent_experiments: Optional[Annotated[int, Field(gt=0)]] = None
    n_top_experiments: Optional[Annotated[int, Field(gt=0)]] = None
    system_prompt: Optional[str] = None

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
