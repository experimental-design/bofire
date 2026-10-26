"""Minimal repro: DoE with a mixture (sum == 1) and an NChooseK constraint.

All four inputs are in (0, 0.6) and at most two may be active. A feasible design
exists (e.g. 0.6 + 0.4 + 0 + 0), but ``ask`` raises ``ConstraintNotFulfilledError``
because the NChooseK patterns also pin rows to a single active variable, which can
never sum to 1 when its upper bound is 0.6.
"""

import warnings

import numpy as np

from bofire.data_models.constraints.api import (
    LinearEqualityConstraint,
    NChooseKConstraint,
)
from bofire.data_models.domain.api import Domain
from bofire.data_models.features.api import ContinuousInput, ContinuousOutput
from bofire.data_models.strategies.api import DoEStrategy as DoEDataModel
from bofire.data_models.strategies.doe import DOptimalityCriterion
from bofire.strategies.api import DoEStrategy
from bofire.strategies.doe.utils import nchoosek_constraints_as_bounds


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

# Upper bounds per experiment row; rows with one nonzero 0.6 cannot sum to 1.
print(np.array(nchoosek_constraints_as_bounds(domain, 6)).reshape(6, 4, 2)[:, :, 1])

strategy = DoEStrategy(
    data_model=DoEDataModel(
        domain=domain, criterion=DOptimalityCriterion(formula="linear")
    )
)
strategy.ask(6)
