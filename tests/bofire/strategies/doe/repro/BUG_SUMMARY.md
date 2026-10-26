# Mixture (linear equality) + NChooseK: infeasible candidates

Domain: 4 inputs in (0, 0.6) (or (0.1, 0.6) with `allow_zero=True`), sum == 1,
NChooseK max 2 (`min_count=0`). Feasible, e.g. 0.6 + 0.4 + 0 + 0; every feasible
point has exactly two (or, with max 3, two or three) active variables. Constructing
the `Domain` works; the failure appears only in `strategy.ask()`.

| Strategy | Result | Cause |
|---|---|---|
| `DoEStrategy` | `ConstraintNotFulfilledError` | NChooseK patterns are not checked against the mixture constraint |
| `RandomStrategy` | `ValueError` (bound exceeded) | pinned-to-zero features collapse the equality to `x == 1`, bounds ignored |
| `SoboStrategy`, `MoboStrategy` | OK in all tested variants | own pruning path (`_nchoosek_pruning.py`) |

## DoE
`_constraint_patterns` (`bofire/strategies/doe/utils.py`) enumerates active counts
from `max(min_count, 1)` to `max_count`, so a one-active-variable pattern is legal.
`nchoosek_constraints_as_bounds` pins the other variables to `[0, 0]` per experiment
row, but a single variable capped at 0.6 can never sum to 1. The solver cannot repair
this, so `ask` fails at `validate_candidates`.

## RandomStrategy
`_sample_with_nchooseks` pins unselected features to `[0, 0]` and expects an
`InfeasibilityError` for infeasible subsets. But `get_linear_constraints` drops the
pinned features, leaving a single-feature equality; `_sample_from_polytope` treats it
as a "pseudo fixed" feature (`x = rhs / coef = 1.0`) without checking the bounds, so no
error is raised and out-of-bound rows are returned. With `min_count=0` the empty
subset can also yield an all-zero row (sum 0).

## Why it shows up late
`Domain` only validates each constraint alone; nothing checks joint feasibility. The
first check is `validate_candidates` at the end of `ask`.

## Fix directions (not implemented)
- DoE: drop patterns that cannot satisfy the linear equalities given the bounds.
- Random: check pseudo-fixed values against bounds (raise `InfeasibilityError`) and
  skip the empty subset when an equality needs a nonzero sum.
- Optional: an early, partial feasibility check for mixture + NChooseK.

Scripts: `repro_doe_mixture_nchoosek.py`, `..._sweep.py`,
`repro_random_strategy_mixture_nchoosek.py`, `repro_bayesian_strategies_mixture_nchoosek.py`.
