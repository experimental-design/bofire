"""LLM-based strategy for candidate proposal using pydantic-ai."""

import asyncio
import json
from typing import Any, Dict, List, Optional, cast

import pandas as pd
from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field, create_model
from pydantic.types import PositiveInt
from typing_extensions import Self

import bofire.data_models.strategies.api as data_models
from bofire.data_models.domain.api import Domain
from bofire.data_models.llm.api import AnyLLMCapability, AnyLLMProvider
from bofire.llm.context import LLMContext
from bofire.strategies.strategy import Strategy, make_strategy


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
        self._capabilities = data_model.capabilities
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
        captures the domain, output schema, system prompt, capabilities, and
        provider model at first access. Per-call inputs (current experiments,
        pending candidates) are passed in via ``LLMContext`` on each
        ``agent.run()``.
        """
        if self._agent is None:
            from pydantic_ai import Agent, ModelRetry

            import bofire.llm.capabilities_mapper as cap_mapper

            proposal_model = _build_proposal_model(self.domain)
            capabilities = [cap_mapper.map(c) for c in self._capabilities]

            agent = Agent(
                self.pydantic_ai_model,
                deps_type=LLMContext,
                system_prompt=self._system_prompt,
                output_type=proposal_model,
                # pydantic-ai v2 unified ``output_retries`` into ``retries``
                # (``AgentRetries = {tools, output}``). We only retry on output
                # validation failures, so map onto the ``output`` budget.
                retries={"output": self._output_retries},
                capabilities=capabilities or None,
                name="LLMStrategy",
            )

            @agent.system_prompt
            async def add_domain_description(ctx) -> str:
                deps: LLMContext = ctx.deps
                return deps.domain.to_description()

            @agent.output_validator
            async def validate_against_domain(ctx, proposal):
                deps: LLMContext = ctx.deps
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

    async def _ask_async(self, candidate_count: int) -> pd.DataFrame:
        """Async implementation of candidate generation."""
        deps = LLMContext(
            domain=self.domain,
            experiments=self.experiments,
            candidates=self.candidates,
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
        capabilities: Optional[List[AnyLLMCapability]] = None,
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
            capabilities: Capabilities to attach to the agent. If omitted, the
                data model default (a single ``ExperimentAccessCapability``) is
                used. Supplying a list replaces the default.
            system_prompt: Custom system prompt override.
            seed: Random seed.

        Returns:
            A new LLMStrategy instance.
        """
        return cast(Self, make_strategy(cls, data_models.LLMStrategy, locals()))
