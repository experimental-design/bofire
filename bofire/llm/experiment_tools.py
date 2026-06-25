"""Experiment-inspection tools exposed to the LLM via a capability.

These build a pydantic-ai ``FunctionToolset`` that lets the agent query the
strategy's experiment table on demand (recent rows, top rows by objective,
nearest neighbours, summary statistics) and list pending candidates, instead
of receiving a pre-rendered dump in the prompt. The tools read live state from
``RunContext[LLMContext]``; the toolset's instructions surface only the counts
of experiments and pending candidates to motivate tool use.

The per-query logic lives in module-level pure functions that take an
``LLMContext`` directly (so they are unit-testable without a ``RunContext``);
the toolset wraps them in thin ``RunContext`` adapters.
"""

import json
from typing import Optional

import numpy as np
import pandas as pd
from pydantic_ai import FunctionToolset, RunContext

from bofire.data_models.domain.api import Domain
from bofire.data_models.objectives.api import MinimizeObjective
from bofire.llm.context import LLMContext


def _objective(domain: Domain) -> tuple[Optional[str], bool]:
    """Return the single objective output key and whether lower is better.

    The single-objective invariant is enforced at the data-model level, so the
    first output carrying an objective is the relevant one.
    """
    for feat in domain.outputs:
        if feat.objective is not None:
            return feat.key, isinstance(feat.objective, MinimizeObjective)
    return None, False


def _display_cols(domain: Domain, df: pd.DataFrame) -> list[str]:
    """Input + output columns that are actually present in the frame."""
    keys = domain.inputs.get_keys() + domain.outputs.get_keys()
    return [k for k in keys if k in df.columns]


def _records(df: pd.DataFrame, cols: list[str]) -> list[dict]:
    """JSON-native records (avoids numpy types leaking into tool output)."""
    return json.loads(df[cols].to_json(orient="records"))


def _experiments(deps: LLMContext) -> Optional[pd.DataFrame]:
    df = deps.experiments
    if df is None or len(df) == 0:
        return None
    return df


# --- pure query functions (unit-testable with a plain LLMContext) ---


def recent_experiments(deps: LLMContext, n: int, max_rows: int) -> list[dict]:
    """The n most recent experiments (most recently added last)."""
    df = _experiments(deps)
    if df is None:
        return []
    cols = _display_cols(deps.domain, df)
    return _records(df.tail(min(n, max_rows)), cols)


def top_experiments(deps: LLMContext, n: int, max_rows: int) -> list[dict]:
    """The top n experiments ranked by the objective (best first)."""
    df = _experiments(deps)
    if df is None:
        return []
    cols = _display_cols(deps.domain, df)
    key, ascending = _objective(deps.domain)
    if key is None or key not in df.columns:
        return _records(df.head(min(n, max_rows)), cols)
    ranked = df.sort_values(by=key, ascending=ascending)
    return _records(ranked.head(min(n, max_rows)), cols)


def near_experiments(
    deps: LLMContext, point: dict, k: int, max_rows: int
) -> list[dict]:
    """The k experiments closest to *point* (Euclidean over numeric inputs)."""
    df = _experiments(deps)
    if df is None:
        return []
    cols = _display_cols(deps.domain, df)
    numeric = [
        c
        for c in deps.domain.inputs.get_keys()
        if c in point and c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not numeric:
        return _records(df.head(min(k, max_rows)), cols)
    target = np.array([float(point[c]) for c in numeric])
    dist = np.sqrt(((df[numeric].astype(float).to_numpy() - target) ** 2).sum(axis=1))
    idx = np.argsort(dist)[: min(k, max_rows)]
    return _records(df.iloc[idx], cols)


def experiment_summary(deps: LLMContext) -> dict:
    """Per-feature numeric statistics and the best objective value."""
    df = _experiments(deps)
    if df is None:
        return {"n_experiments": 0}
    cols = _display_cols(deps.domain, df)
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    out: dict = {"n_experiments": int(len(df))}
    if numeric_cols:
        out["statistics"] = json.loads(df[numeric_cols].describe().to_json())
    key, ascending = _objective(deps.domain)
    if key is not None and key in df.columns:
        best = df[key].min() if ascending else df[key].max()
        out["best_objective"] = {
            "key": key,
            "value": float(best),
            "direction": "minimize" if ascending else "maximize",
        }
    return out


def pending_candidates(deps: LLMContext, max_rows: int) -> list[dict]:
    """Candidates already proposed but not yet evaluated (avoid duplicates)."""
    df = deps.candidates
    if df is None or len(df) == 0:
        return []
    cols = [k for k in deps.domain.inputs.get_keys() if k in df.columns]
    return _records(df.head(max_rows), cols)


def build_experiment_toolset(max_rows: int) -> FunctionToolset:
    """Build the experiment-access toolset, capping rows per call at *max_rows*."""

    def _instructions(ctx: RunContext[LLMContext]) -> str:
        n = 0 if ctx.deps.experiments is None else len(ctx.deps.experiments)
        m = 0 if ctx.deps.candidates is None else len(ctx.deps.candidates)
        return (
            f"There are {n} completed experiment(s) and {m} pending candidate(s). "
            f"Use the experiment tools to inspect them before proposing candidates. "
            f"Each tool returns at most {max_rows} rows."
        )

    def inspect_recent(ctx: RunContext[LLMContext], n: int = 10) -> list[dict]:
        """Return the n most recent experiments (most recently added last)."""
        return recent_experiments(ctx.deps, n, max_rows)

    def inspect_top(ctx: RunContext[LLMContext], n: int = 10) -> list[dict]:
        """Return the top n experiments ranked by the objective (best first)."""
        return top_experiments(ctx.deps, n, max_rows)

    def inspect_near(
        ctx: RunContext[LLMContext], point: dict, k: int = 5
    ) -> list[dict]:
        """Return the k experiments closest to *point* over numeric inputs."""
        return near_experiments(ctx.deps, point, k, max_rows)

    def summary_stats(ctx: RunContext[LLMContext]) -> dict:
        """Return per-feature numeric statistics and the best objective value."""
        return experiment_summary(ctx.deps)

    def list_pending_candidates(ctx: RunContext[LLMContext]) -> list[dict]:
        """Return candidates already proposed but not yet evaluated."""
        return pending_candidates(ctx.deps, max_rows)

    return FunctionToolset(
        tools=[
            inspect_recent,
            inspect_top,
            inspect_near,
            summary_stats,
            list_pending_candidates,
        ],
        instructions=_instructions,
    )
