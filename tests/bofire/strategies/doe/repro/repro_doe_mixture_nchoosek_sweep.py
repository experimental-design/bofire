"""Sweep bounds / allow_zero / max_count for mixture + NChooseK DoE (see the minimal repro)."""

import warnings

from bofire.data_models.constraints.api import (
    LinearEqualityConstraint,
    NChooseKConstraint,
)
from bofire.data_models.domain.api import Domain
from bofire.data_models.features.api import ContinuousInput, ContinuousOutput
from bofire.data_models.strategies.api import DoEStrategy as DoEDataModel
from bofire.data_models.strategies.doe import DOptimalityCriterion
from bofire.strategies.api import DoEStrategy


warnings.filterwarnings("ignore")


def run(bounds, allow_zero, max_count, n=6):
    keys = [f"x{i}" for i in range(4)]
    domain = Domain.from_lists(
        inputs=[
            ContinuousInput(key=k, bounds=bounds, allow_zero=allow_zero) for k in keys
        ],
        outputs=[ContinuousOutput(key="o")],
        constraints=[
            LinearEqualityConstraint(features=keys, coefficients=[1] * 4, rhs=1),
            NChooseKConstraint(
                features=keys, min_count=0, max_count=max_count, none_also_valid=False
            ),
        ],
    )
    strategy = DoEStrategy(
        data_model=DoEDataModel(
            domain=domain, criterion=DOptimalityCriterion(formula="linear")
        )
    )
    try:
        x = strategy.ask(n)
        print(bounds, allow_zero, max_count, "OK")
        print(x[keys].assign(sum=x[keys].sum(axis=1)).round(3))
    except Exception as e:
        print(bounds, allow_zero, max_count, "FAIL", type(e).__name__)
        print(str(e)[-700:])


for case in [
    ((0, 0.6), False, 2),
    ((0.1, 0.6), True, 2),
    ((0, 0.6), False, 3),
    ((0.1, 0.6), True, 3),
    ((0, 1), False, 2),
]:
    run(*case)
