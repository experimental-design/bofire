"""LLM-based strategy for candidate proposal using pydantic-ai."""

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, cast

import pandas as pd
from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field, create_model
from pydantic.types import PositiveInt
from typing_extensions import Self

import bofire.data_models.strategies.api as data_models
from bofire.data_models.domain.api import Domain
from bofire.data_models.features.api import ContinuousOutput
from bofire.data_models.llm.api import AnyLLMProvider
from bofire.data_models.objectives.api import MinimizeObjective
from bofire.strategies.strategy import Strategy, make_strategy


# --- Dependencies dataclass for pydantic-ai agent ---
@dataclass
class _LLMDeps:
    domain: Domain
    experiments_text: str


# --- Default system prompt ---
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


# --- Experiment selection ---
def _select_experiments(
    experiments: pd.DataFrame,
    domain: Domain,
    n_recent: Optional[int],
    n_top: Optional[int],
) -> tuple[pd.DataFrame, str]:
    """Select a subset of experiments for LLM presentation.

    Returns a tuple of (selected_df, description_text) where description_text
    explains to the LLM what kind of experiments are being shown.
    """
    if n_recent is None and n_top is None:
        return experiments, f"All {len(experiments)} experiments"

    parts = []
    desc_parts = []

    if n_recent is not None:
        recent = experiments.tail(n_recent)
        parts.append(recent)
        desc_parts.append(f"last {len(recent)} most recent")

    if n_top is not None:
        # Use the single objective output (validated at data model level)
        for feat in domain.outputs:
            if isinstance(feat, ContinuousOutput) and feat.objective is not None:
                metric_key = feat.key
                ascending = isinstance(feat.objective, MinimizeObjective)
                break

        sorted_exps = experiments.sort_values(by=metric_key, ascending=ascending)
        top = sorted_exps.head(n_top)
        parts.append(top)
        direction = "lowest" if ascending else "highest"
        desc_parts.append(f"top {len(top)} by {direction} {metric_key}")

    selected = pd.concat(parts).drop_duplicates()
    description = (
        " + ".join(desc_parts)
        + f" ({len(selected)} unique shown out of {len(experiments)} total)"
    )
    return selected, description


def _reasoning_column_name(domain: Domain) -> str:
    """Pick a column name for the LLM reasoning that does not collide with any
    feature key in the domain. Appends trailing underscores until unique.
    """
    taken = set(domain.inputs.get_keys()) | set(domain.outputs.get_keys())
    name = "reasoning"
    while name in taken:
        name += "_"
    return name


def _build_proposal_model(domain: Domain) -> type[PydanticBaseModel]:
    """Build the pydantic-ai output model from the Domain."""
    CandidatePoint = domain.inputs.to_pydantic_model()

    class Candidate(PydanticBaseModel):
        """A candidate with its feature values and reasoning."""

        values: CandidatePoint  # type: ignore[valid-type]
        reasoning: str = Field(
            description="Brief explanation of why this candidate was chosen."
        )

    return create_model(
        "CandidateProposal",
        __doc__="A set of proposed candidates with overall reasoning.",
        candidates=(list[Candidate], ...),
        strategy_summary=(
            str,
            Field(description="Overall strategy for the proposed candidates."),
        ),
    )


# --- Strategy ---
class LLMStrategy(Strategy):
    """Strategy that uses an LLM to propose optimization candidates.

    Uses pydantic-ai structured output with a dynamically generated schema
    that matches the Domain's input features. Domain validation catches
    constraint violations, and pydantic-ai retries automatically.
    """

    def __init__(self, data_model: data_models.LLMStrategy, **kwargs):
        super().__init__(data_model=data_model, **kwargs)
        self._llm_provider = data_model.llm
        self._output_retries = data_model.output_retries
        self._n_recent_experiments = data_model.n_recent_experiments
        self._n_top_experiments = data_model.n_top_experiments
        self._system_prompt = data_model.system_prompt or _DEFAULT_SYSTEM_PROMPT
        self._pydantic_ai_model = None
        self._agent = None

    @property
    def pydantic_ai_model(self):
        """Lazily constructed pydantic-ai model. Built once on first access.

        Kept out of ``__init__`` so that instantiating an ``LLMStrategy`` does
        not resolve provider environment variables (e.g. API keys).
        """
        if self._pydantic_ai_model is None:
            import bofire.llm.mapper as llm_mapper

            self._pydantic_ai_model = llm_mapper.map(self._llm_provider)
        return self._pydantic_ai_model

    @property
    def agent(self):
        """Lazily constructed pydantic-ai Agent. Built once on first access.

        Per BoFire's "build once, execute many" philosophy: the Agent
        captures the domain, output schema, system prompt, and provider
        model at first access. Per-call inputs (current experiments,
        candidate count) are passed in via ``_LLMDeps`` on each
        ``agent.run()``.
        """
        if self._agent is None:
            from pydantic_ai import Agent, ModelRetry

            proposal_model = _build_proposal_model(self.domain)

            agent = Agent(
                self.pydantic_ai_model,
                system_prompt=self._system_prompt,
                output_type=proposal_model,
                output_retries=self._output_retries,
                name="LLMStrategy",
            )

            @agent.system_prompt
            async def add_domain_description(ctx) -> str:
                deps: _LLMDeps = ctx.deps
                parts = [deps.domain.to_description()]
                if deps.experiments_text:
                    parts.append(deps.experiments_text)
                return "\n".join(parts)

            @agent.output_validator
            async def validate_against_domain(ctx, proposal):
                deps: _LLMDeps = ctx.deps
                rows = [c.values.model_dump() for c in proposal.candidates]
                candidates_df = pd.DataFrame(rows)
                try:
                    deps.domain.validate_candidates(candidates_df, only_inputs=True)
                except Exception as e:
                    candidates_json = json.dumps(rows, indent=2)
                    # ModelRetry (not ValueError) is the only exception
                    # pydantic-ai catches in output validators to trigger a
                    # retry within ``output_retries``. See
                    # pydantic_ai/_result.py.
                    raise ModelRetry(
                        f"Candidate validation failed: {e}\n\n"
                        f"The proposed candidates were:\n{candidates_json}\n\n"
                        f"Please fix the candidates to satisfy all constraints "
                        f"and feature bounds, then try again."
                    ) from e
                return proposal

            self._agent = agent
        return self._agent

    def has_sufficient_experiments(self) -> bool:
        """LLM can propose candidates with zero experiments (cold start)."""
        return True

    def _ask(self, candidate_count: Optional[PositiveInt] = None) -> pd.DataFrame:
        """Generate candidates by calling the LLM.

        Bridges async pydantic-ai into sync BoFire via asyncio.run().
        """
        if candidate_count is None:
            candidate_count = 1
        return asyncio.run(self._ask_async(candidate_count))

    def _build_experiments_text(self) -> str:
        """Render the currently held experiments as a JSON block for the LLM."""
        if self.experiments is None or len(self.experiments) == 0:
            return ""
        selected, description = _select_experiments(
            self.domain.outputs.preprocess_experiments_all_valid_outputs(
                self.experiments
            ),
            self.domain,
            self._n_recent_experiments,
            self._n_top_experiments,
        )
        display_cols = self.domain.inputs.get_keys() + self.domain.outputs.get_keys()
        experiments_json = json.dumps(
            selected[display_cols].to_dict(orient="records"), indent=2
        )
        return (
            f"\n## Previous Experiments ({description})\n"
            f"```json\n{experiments_json}\n```"
        )

    async def _ask_async(self, candidate_count: int) -> pd.DataFrame:
        """Async implementation of candidate generation."""
        deps = _LLMDeps(
            domain=self.domain,
            experiments_text=self._build_experiments_text(),
        )

        result = await self.agent.run(
            f"Propose {candidate_count} diverse candidate points for this optimization problem.",
            deps=deps,
            model_settings=self._data_model.model_settings,
        )

        proposal = result.output
        reasoning_col = _reasoning_column_name(self.domain)
        rows = [
            {**c.values.model_dump(), reasoning_col: c.reasoning}
            for c in proposal.candidates
        ]
        return pd.DataFrame(rows)

    @classmethod
    def make(
        cls,
        domain: Domain,
        llm: AnyLLMProvider,
        model_settings: Optional[Dict[str, Any]] = None,
        output_retries: Optional[int] = None,
        n_recent_experiments: Optional[int] = None,
        n_top_experiments: Optional[int] = None,
        system_prompt: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Self:
        """Create a new LLMStrategy instance.

        Args:
            domain: The optimization domain.
            llm: LLM provider configuration.
            model_settings: Optional dict forwarded to pydantic-ai's
                ``model_settings`` (e.g. ``{"temperature": 0.2,
                "max_tokens": 4096, "thinking": "high"}``).
            output_retries: Number of retries for output validation.
            n_recent_experiments: Number of recent experiments to show.
            n_top_experiments: Number of top experiments to show.
            system_prompt: Custom system prompt override.
            seed: Random seed.

        Returns:
            A new LLMStrategy instance.
        """
        return cast(Self, make_strategy(cls, data_models.LLMStrategy, locals()))
