from typing import Annotated, Literal, Optional, Type, Union

from pydantic import Field, model_validator

from bofire.data_models.constraints.api import (
    Constraint,
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
)
from bofire.data_models.features.api import ContinuousOutput, Feature
from bofire.data_models.llm.api import AnyLLMProvider
from bofire.data_models.objectives.api import (
    MaximizeObjective,
    MinimizeObjective,
    Objective,
)
from bofire.data_models.strategies.strategy import Strategy


ThinkingLevel = Union[bool, Literal["minimal", "low", "medium", "high", "xhigh"]]


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
        temperature: Sampling temperature for the LLM.
        max_tokens: Maximum number of tokens in the LLM response.
        thinking: Reasoning effort level for the LLM.
        n_recent_experiments: Number of most recent experiments to show the LLM.
        n_top_experiments: Number of top-performing experiments to show the LLM.
        top_metric_key: Output feature key to rank experiments by for top-N selection.
        system_prompt: Optional override for the default system prompt.
    """

    type: Literal["LLMStrategy"] = "LLMStrategy"

    llm: AnyLLMProvider
    temperature: Optional[Annotated[float, Field(ge=0.0, le=2.0)]] = None
    max_tokens: Optional[Annotated[int, Field(gt=0)]] = None
    thinking: Optional[ThinkingLevel] = None
    n_recent_experiments: Optional[Annotated[int, Field(gt=0)]] = None
    n_top_experiments: Optional[Annotated[int, Field(gt=0)]] = None
    top_metric_key: Optional[str] = None
    system_prompt: Optional[str] = None

    @model_validator(mode="after")
    def validate_single_objective(self):
        """Validate that the domain has exactly one output with a supported objective."""
        outputs_with_obj = [
            f
            for f in self.domain.outputs
            if isinstance(f, ContinuousOutput) and f.objective is not None
        ]
        if len(outputs_with_obj) != 1:
            raise ValueError(
                f"LLMStrategy requires exactly one output with an objective, "
                f"got {len(outputs_with_obj)}."
            )
        obj = outputs_with_obj[0].objective
        if not isinstance(obj, (MaximizeObjective, MinimizeObjective)):
            raise ValueError(
                f"LLMStrategy only supports MaximizeObjective or MinimizeObjective, "
                f"got {type(obj).__name__}."
            )
        return self

    @model_validator(mode="after")
    def validate_top_metric_key(self):
        """Validate that top_metric_key references a valid output feature."""
        if self.top_metric_key is not None:
            keys = self.domain.outputs.get_keys()
            if self.top_metric_key not in keys:
                raise ValueError(
                    f"top_metric_key '{self.top_metric_key}' is not a valid output "
                    f"feature key. Available: {keys}"
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

    @classmethod
    def is_objective_implemented(cls, my_type: Type[Objective]) -> bool:
        return my_type in [MaximizeObjective, MinimizeObjective]
