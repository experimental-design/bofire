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
from bofire.strategies.llm import _build_proposal_model


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


# --- experiment-access tools ---


def _ctx(domain, experiments=None, candidates=None):
    from bofire.llm.context import LLMContext

    return LLMContext(domain=domain, experiments=experiments, candidates=candidates)


def test_recent_experiments(simple_domain, experiments):
    from bofire.llm.experiment_tools import recent_experiments

    rows = recent_experiments(_ctx(simple_domain, experiments), n=3, max_rows=50)
    assert [r["y"] for r in rows] == [80, 90, 100]
    # output columns included, helper/valid columns excluded
    assert set(rows[0]) == {"x1", "x2", "x3", "y"}


def test_recent_experiments_capped(simple_domain, experiments):
    from bofire.llm.experiment_tools import recent_experiments

    rows = recent_experiments(_ctx(simple_domain, experiments), n=100, max_rows=4)
    assert len(rows) == 4


def test_top_experiments_maximize(simple_domain, experiments):
    from bofire.llm.experiment_tools import top_experiments

    rows = top_experiments(_ctx(simple_domain, experiments), n=3, max_rows=50)
    assert {r["y"] for r in rows} == {80, 90, 100}


def test_top_experiments_minimize(experiments):
    from bofire.llm.experiment_tools import top_experiments

    domain = Domain.from_lists(
        inputs=[
            ContinuousInput(key="x1", bounds=(0, 1)),
            ContinuousInput(key="x2", bounds=(0, 1)),
            CategoricalInput(key="x3", categories=["a", "b", "c"]),
        ],
        outputs=[ContinuousOutput(key="y", objective=MinimizeObjective(w=1.0))],
    )
    rows = top_experiments(_ctx(domain, experiments), n=3, max_rows=50)
    assert {r["y"] for r in rows} == {10, 20, 30}


def test_near_experiments(simple_domain, experiments):
    from bofire.llm.experiment_tools import near_experiments

    rows = near_experiments(
        _ctx(simple_domain, experiments), point={"x1": 0.1, "x2": 0.9}, k=1, max_rows=50
    )
    assert len(rows) == 1
    assert rows[0]["y"] == 10  # the (0.1, 0.9) row


def test_experiment_summary(simple_domain, experiments):
    from bofire.llm.experiment_tools import experiment_summary

    summary = experiment_summary(_ctx(simple_domain, experiments))
    assert summary["n_experiments"] == 10
    assert summary["best_objective"] == {
        "key": "y",
        "value": 100.0,
        "direction": "maximize",
    }


def test_pending_candidates(simple_domain):
    from bofire.llm.experiment_tools import pending_candidates

    cands = pd.DataFrame({"x1": [0.5], "x2": [0.5], "x3": ["a"]})
    rows = pending_candidates(_ctx(simple_domain, candidates=cands), max_rows=50)
    assert rows == [{"x1": 0.5, "x2": 0.5, "x3": "a"}]


def test_experiment_tools_empty(simple_domain):
    from bofire.llm.experiment_tools import (
        experiment_summary,
        pending_candidates,
        recent_experiments,
    )

    ctx = _ctx(simple_domain)
    assert recent_experiments(ctx, n=5, max_rows=50) == []
    assert pending_candidates(ctx, max_rows=50) == []
    assert experiment_summary(ctx) == {"n_experiments": 0}


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


def test_llm_strategy_ask_with_experiments_and_tools():
    """ask() succeeds with experiments told and the default ExperimentAccessCapability.

    TestModel exercises every available tool, so this also confirms the
    experiment-access toolset is wired and callable end-to-end.
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
    assert len(data_model.capabilities) == 1  # default ExperimentAccessCapability
    strategy = LLMStrategy(data_model=data_model)
    strategy._pydantic_ai_model = TestModel()

    experiments = pd.DataFrame({"x1": [1.0, 2.0], "x2": [3.0, 4.0], "y": [10.0, 20.0]})
    strategy.tell(domain.validate_experiments(experiments))

    candidates = strategy.ask(1)
    assert len(candidates) == 1
    assert "reasoning" in candidates.columns
