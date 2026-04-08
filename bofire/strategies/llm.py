"""LLM-based strategy for candidate proposal using pydantic-ai."""

import asyncio
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field, create_model
from pydantic.types import PositiveInt

import bofire.data_models.strategies.api as data_models
from bofire.data_models.domain.api import Domain
from bofire.data_models.features.api import ContinuousOutput
from bofire.data_models.objectives.api import MinimizeObjective
from bofire.strategies.strategy import Strategy


# --- Dependencies dataclass for pydantic-ai agent ---
@dataclass
class _LLMDeps:
    domain: Domain
    n_candidates: int
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


# --- Proposal model builder ---
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
        self._temperature = data_model.temperature
        self._max_tokens = data_model.max_tokens
        self._thinking = data_model.thinking
        self._n_recent_experiments = data_model.n_recent_experiments
        self._n_top_experiments = data_model.n_top_experiments
        self._system_prompt = data_model.system_prompt or _DEFAULT_SYSTEM_PROMPT

        # Build the pydantic-ai model at init (LLM connection doesn't change)
        import bofire.llm.mapper as llm_mapper

        self._pydantic_ai_model = llm_mapper.map(self._llm_provider)

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

    async def _ask_async(self, candidate_count: int) -> pd.DataFrame:
        """Async implementation of candidate generation."""
        from pydantic_ai import Agent

        # Build output schema fresh each call (domain may have changed)
        proposal_model = _build_proposal_model(self.domain)

        # Create agent
        agent = Agent(
            self._pydantic_ai_model,
            system_prompt=self._system_prompt,
            output_type=proposal_model,
        )

        # Add domain description as dynamic system prompt
        @agent.system_prompt
        async def add_domain_description(ctx) -> str:
            deps: _LLMDeps = ctx.deps
            parts = [deps.domain.to_description()]
            if deps.experiments_text:
                parts.append(deps.experiments_text)
            return "\n".join(parts)

        # Add output validator: domain.validate_candidates with verbose errors
        @agent.output_validator
        async def validate_against_domain(ctx, proposal):
            deps: _LLMDeps = ctx.deps
            rows = [c.values.model_dump() for c in proposal.candidates]
            candidates_df = pd.DataFrame(rows)
            try:
                deps.domain.validate_candidates(candidates_df, only_inputs=True)
            except Exception as e:
                raise ValueError(
                    f"Candidate validation failed: {e}\n\n"
                    f"The proposed candidates were:\n{candidates_df.to_string()}\n\n"
                    f"Please fix the candidates to satisfy all constraints and "
                    f"feature bounds, then try again."
                ) from e
            return proposal

        # Prepare experiment text
        experiments_text = ""
        if self.experiments is not None and len(self.experiments) > 0:
            selected, description = _select_experiments(
                self.experiments,
                self.domain,
                self._n_recent_experiments,
                self._n_top_experiments,
            )
            experiments_text = (
                f"\n## Previous Experiments ({description})\n"
                f"{selected.to_string(index=False)}"
            )

        deps = _LLMDeps(
            domain=self.domain,
            n_candidates=candidate_count,
            experiments_text=experiments_text,
        )

        # Build ModelSettings
        model_settings = {}
        if self._temperature is not None:
            model_settings["temperature"] = self._temperature
        if self._max_tokens is not None:
            model_settings["max_tokens"] = self._max_tokens
        if self._thinking is not None:
            model_settings["thinking"] = self._thinking

        result = await agent.run(
            f"Propose {candidate_count} diverse candidate points for this optimization problem.",
            deps=deps,
            model_settings=model_settings if model_settings else None,
        )

        proposal = result.output
        return pd.DataFrame([c.values.model_dump() for c in proposal.candidates])
