"""RandomStrategy + mixture (sum == 1) + NChooseK(max 2 of 4), inputs in (0, 0.6).

Run:  python repro_random_strategy_mixture_nchoosek.py

Domain: x0..x3 in [0, 0.6], x0+x1+x2+x3 == 1, at most 2 of them nonzero.
A feasible point exists (0.6 + 0.4 + 0 + 0), and every feasible point has
EXACTLY two active variables (one variable alone is capped at 0.6 < 1).

What goes wrong
---------------
``RandomStrategy._sample_with_nchooseks`` draws a set of active features,
pins every other feature to bounds [0, 0], and samples the remaining polytope.
When the drawn set has ONE active feature the reduced problem is infeasible
(x == 1 with x <= 0.6) and the code is meant to drop it (it catches
``InfeasibilityError``). It never gets that far:

  1. ``get_linear_constraints`` strips pinned features out of the equality, so
     the mixture constraint becomes the single-feature row ``x2 == 1``.
  2. ``_sample_from_polytope`` treats a single-feature equality as a
     "pseudo fixed" feature and sets ``fixed_features[x2] = rhs / coef = 1.0``
     without comparing it to x2's bounds.

The sampler therefore returns rows like [0, 0, 1.0, 0]; ``ask`` then fails in
``validate_candidates`` with "not all values of input feature ... are smaller
than upper bound 0.6".
"""

import warnings

from bofire.data_models.constraints.api import (
    LinearEqualityConstraint,
    NChooseKConstraint,
)
from bofire.data_models.domain.api import Domain
from bofire.data_models.features.api import ContinuousInput, ContinuousOutput
from bofire.data_models.strategies.api import RandomStrategy as RandomDataModel
from bofire.strategies.api import RandomStrategy
from bofire.utils.torch_tools import get_linear_constraints


warnings.filterwarnings("ignore")

keys = [f"x{i}" for i in range(4)]
domain = Domain.from_lists(
    inputs=[ContinuousInput(key=k, bounds=(0, 0.6)) for k in keys],
    outputs=[ContinuousOutput(key="o")],
    constraints=[
        LinearEqualityConstraint(features=keys, coefficients=[1] * 4, rhs=1),
        NChooseKConstraint(
            features=keys, min_count=0, max_count=2, none_also_valid=False
        ),
    ],
)

# --- Step 1: show the constraint collapsing to a single-feature equality -----
# Pin x0, x1, x3 to zero exactly as ``_sample_with_nchooseks`` does.
pinned = domain.model_copy(deep=True)
for i in (0, 1, 3):
    pinned.inputs.features[i].bounds = [0.0, 0.0]
print("Equality rows after pinning x0, x1, x3 to 0:")
print(get_linear_constraints(pinned, LinearEqualityConstraint, unit_scaled=False))
print("  -> single feature (index 2) with rhs -1 / coef -1: treated as x2 == 1.0\n")

# --- Step 2: the sampler returns the out-of-bounds rows ----------------------
strategy = RandomStrategy(data_model=RandomDataModel(domain=domain))
candidates = strategy._ask(6)  # bypass validation to see the raw samples
candidates["sum"] = candidates[keys].sum(axis=1)
print("Raw samples (rows with a single 1.0 exceed the bound 0.6; an all-zero row\n"
      "also appears when min_count=0 lets the EMPTY subset through, sum = 0):")
print(candidates[keys + ["sum"]].round(4).to_string())

# --- Step 3: the public API fails ---------------------------------------------
print("\nstrategy.ask(6):")
strategy.ask(6)  # raises ValueError: ... smaller than upper bound `0.6`
