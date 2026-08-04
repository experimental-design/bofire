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
    """Strategy that uses a large language model to propose optimization candidates.

    Instead of fitting a surrogate and optimizing an acquisition function,
    this strategy lets an LLM read the optimization problem — feature bounds,
    constraints, objectives, contextual descriptions, and prior experiments —
    and directly propose candidate points. It is useful for cold-start
    designs, mixed / categorical / molecular spaces where domain knowledge
    helps, and exploration informed by written context (``Feature.context``
    and ``Domain.context``).

    It is not a replacement for a Bayesian optimizer on well-understood
    numerical problems: there is no calibrated uncertainty model and no
    acquisition function. Treat candidates as informed heuristics, not
    optima.

    On each ``ask()``, a pydantic output schema is generated from the
    domain's input features and the LLM is prompted with a textual problem
    description plus, optionally, a selection of prior experiments. Returned
    candidates are validated against the domain; bound or constraint
    violations are sent back to the LLM as retry messages via pydantic-ai's
    ``output_retries``.

    Currently supports single-objective optimization with ``MaximizeObjective``
    or ``MinimizeObjective``, and ``LinearEquality``, ``LinearInequality``,
    and ``NChooseK`` constraints. All feature types are supported.

    Example:
        Basic usage::

            strategy = LLMStrategy.make(
                domain=domain,
                llm=AnthropicLLMProvider(model="claude-sonnet-4-5"),
            )

        Enable extended reasoning for harder problems (many constraints,
        rich context). ``thinking`` is pydantic-ai's cross-provider
        capability key — it maps to Anthropic's extended thinking, OpenAI's
        ``reasoning_effort``, and similar mechanisms on other providers.
        Reasoning increases cost and latency considerably, so it is not
        enabled by default::

            strategy = LLMStrategy.make(
                domain=domain,
                llm=AnthropicLLMProvider(model="claude-sonnet-4-5"),
                model_settings={"thinking": "high"},
            )

    Attributes:
        llm: LLM provider configuration.
        model_settings: Optional dict forwarded directly to pydantic-ai's
            ``model_settings``. Useful keys include ``temperature``,
            ``max_tokens``, ``top_p``, ``seed``, ``timeout``, and the
            cross-provider capability ``thinking`` (``"low"`` / ``"medium"``
            / ``"high"``). Provider-prefixed keys such as
            ``anthropic_thinking`` or ``openai_reasoning_effort`` are also
            accepted as escape hatches for finer control. Keys are not
            validated by BoFire — pydantic-ai and the underlying provider
            SDK are the source of truth.
        output_retries: Number of retries when output validation fails
            (constraint or bound violations). Each retry sends the LLM the
            invalid candidates and the error so it can correct.
        n_recent_experiments: If set, only the most recent N experiments
            are shown to the LLM. Keeps prompt size bounded on long
            campaigns.
        n_top_experiments: If set, the top N experiments by objective
            value are shown to the LLM. Combine with
            ``n_recent_experiments`` to mix recency and quality.
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
