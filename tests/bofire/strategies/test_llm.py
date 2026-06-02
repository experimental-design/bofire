"""Tests for LLMStrategy utility functions and data model integration."""

import importlib.util

import pandas as pd
import pytest

from bofire.data_models.domain.api import Domain
from bofire.data_models.features.api import (
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
)
from bofire.data_models.llm.provider import AnthropicLLMProvider
from bofire.data_models.objectives.api import MaximizeObjective, MinimizeObjective
from bofire.data_models.strategies.api import LLMStrategy as LLMStrategyDataModel
from bofire.strategies.api import LLMStrategy
from bofire.strategies.llm import _build_proposal_model, _select_experiments


PYDANTIC_AI_AVAILABLE = importlib.util.find_spec("pydantic_ai") is not None

pytestmark = pytest.mark.skipif(
    not PYDANTIC_AI_AVAILABLE,
    reason="requires pydantic-ai (install with [llm] extra)",
)


# --- Fixtures ---


@pytest.fixture()
def simple_domain():
    return Domain.from_lists(
        inputs=[
            ContinuousInput(key="x1", bounds=(0, 1)),
            ContinuousInput(key="x2", bounds=(0, 1)),
            CategoricalInput(key="x3", categories=["a", "b", "c"]),
        ],
        outputs=[ContinuousOutput(key="y", objective=MaximizeObjective(w=1.0))],
    )


@pytest.fixture()
def experiments():
    return pd.DataFrame(
        {
            "x1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "x2": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
            "x3": ["a", "b", "c", "a", "b", "c", "a", "b", "c", "a"],
            "y": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "valid_y": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        }
    )


# --- _select_experiments ---


def test_select_experiments_all(simple_domain, experiments):
    selected, desc = _select_experiments(experiments, simple_domain, None, None)
    assert len(selected) == 10
    assert "All" in desc


def test_select_experiments_recent(simple_domain, experiments):
    selected, desc = _select_experiments(experiments, simple_domain, 3, None)
    assert len(selected) == 3
    assert selected["y"].tolist() == [80, 90, 100]
    assert "recent" in desc


def test_select_experiments_top(simple_domain, experiments):
    selected, desc = _select_experiments(experiments, simple_domain, None, 3)
    assert len(selected) == 3
    assert set(selected["y"].tolist()) == {80, 90, 100}
    assert "top" in desc
    assert "highest" in desc


def test_select_experiments_top_minimize(experiments):
    domain = Domain.from_lists(
        inputs=[
            ContinuousInput(key="x1", bounds=(0, 1)),
            ContinuousInput(key="x2", bounds=(0, 1)),
            CategoricalInput(key="x3", categories=["a", "b", "c"]),
        ],
        outputs=[ContinuousOutput(key="y", objective=MinimizeObjective(w=1.0))],
    )
    selected, desc = _select_experiments(experiments, domain, None, 3)
    assert set(selected["y"].tolist()) == {10, 20, 30}
    assert "lowest" in desc


def test_select_experiments_both_deduplicates(simple_domain, experiments):
    selected, desc = _select_experiments(experiments, simple_domain, 5, 5)
    # Last 5: y=[60,70,80,90,100], Top 5: y=[60,70,80,90,100] — same set
    assert len(selected) == 5
    assert "unique" in desc


def test_select_experiments_both_union(simple_domain, experiments):
    # Recent 2: y=[90,100], Top 2: y=[90,100] — overlap
    selected, _ = _select_experiments(experiments, simple_domain, 2, 2)
    assert len(selected) == 2

    # Recent 3: y=[80,90,100], Top 3: y=[80,90,100]
    selected, _ = _select_experiments(experiments, simple_domain, 3, 3)
    assert len(selected) == 3


# --- _build_proposal_model ---


def test_build_proposal_model_schema(simple_domain):
    Model = _build_proposal_model(simple_domain)
    schema = Model.model_json_schema()
    assert "candidates" in schema["properties"]
    assert "strategy_summary" in schema["properties"]
    # Check nested CandidatePoint has our features
    candidate_schema = schema["$defs"]["CandidatePoint"]
    assert "x1" in candidate_schema["properties"]
    assert "x2" in candidate_schema["properties"]
    assert "x3" in candidate_schema["properties"]


def test_build_proposal_model_validates(simple_domain):
    Model = _build_proposal_model(simple_domain)

    proposal = Model(
        candidates=[
            {
                "values": {"x1": 0.5, "x2": 0.5, "x3": "a"},
                "reasoning": "test",
            }
        ],
        strategy_summary="test",
    )
    assert len(proposal.candidates) == 1


def test_llm_strategy_rejects_multi_objective():
    domain = Domain.from_lists(
        inputs=[ContinuousInput(key="x", bounds=(0, 1))],
        outputs=[
            ContinuousOutput(key="y1", objective=MaximizeObjective(w=1.0)),
            ContinuousOutput(key="y2", objective=MinimizeObjective(w=1.0)),
        ],
    )
    with pytest.raises(ValueError, match="exactly one output"):
        LLMStrategyDataModel(
            domain=domain,
            llm=AnthropicLLMProvider(api_key_env_var="KEY"),
        )


# --- End-to-end smoke test with pydantic-ai TestModel ---


def test_llm_strategy_ask_with_test_model():
    """Smoke test the full ask() pipeline using pydantic-ai's TestModel.

    TestModel auto-generates structured output matching the agent's schema.
    We pick continuous bounds starting at 0 so the generated defaults satisfy
    the domain validator, and avoid constraints for the same reason.
    """
    from pydantic_ai.models.test import TestModel

    domain = Domain.from_lists(
        inputs=[
            ContinuousInput(key="x1", bounds=(0, 10)),
            ContinuousInput(key="x2", bounds=(0, 10)),
        ],
        outputs=[ContinuousOutput(key="y", objective=MaximizeObjective(w=1.0))],
    )
    data_model = LLMStrategyDataModel(
        domain=domain,
        llm=AnthropicLLMProvider(api_key_env_var="UNUSED_KEY"),
    )
    strategy = LLMStrategy(data_model=data_model)
    # Inject TestModel directly to bypass provider/env-var resolution.
    strategy._pydantic_ai_model = TestModel()

    # TestModel generates a single array item by default, so ask(1).
    candidates = strategy.ask(1)
    assert len(candidates) == 1
    assert "reasoning" in candidates.columns
