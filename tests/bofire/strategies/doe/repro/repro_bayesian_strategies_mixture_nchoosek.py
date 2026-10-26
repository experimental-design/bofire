"""Bayesian strategies + mixture (sum == 1) + NChooseK (see the DoE / Random repros).

Run:  python repro_bayesian_strategies_mixture_nchoosek.py

Same domain as the other repros: x0..x3 in [0, 0.6], sum == 1, at most 2
nonzero. This script checks whether the infeasible-pattern problem found in
DoEStrategy / RandomStrategy also hits the BoTorch-based strategies, which
handle NChooseK through their own path (``strategies/predictives``).

Each case tells the strategy a few feasible experiments (every one has exactly
two active variables summing to 1) and then asks for candidates.
Results are printed as OK / FAIL, with each candidate's sum so a violated
mixture constraint is visible.
"""

import warnings

import pandas as pd

from bofire.data_models.constraints.api import (
    LinearEqualityConstraint,
    NChooseKConstraint,
)
from bofire.data_models.domain.api import Domain
from bofire.data_models.features.api import ContinuousInput, ContinuousOutput
from bofire.data_models.strategies.api import MoboStrategy as MoboDataModel
from bofire.data_models.strategies.api import SoboStrategy as SoboDataModel
from bofire.data_models.objectives.api import MaximizeObjective
from bofire.strategies.api import MoboStrategy, SoboStrategy


warnings.filterwarnings("ignore")

keys = [f"x{i}" for i in range(4)]
# Feasible training points: 2 active variables, summing to 1, each <= 0.6.
points = [
    [0.6, 0.4, 0, 0],
    [0.4, 0, 0.6, 0],
    [0, 0.5, 0.5, 0],
    [0, 0, 0.4, 0.6],
    [0.6, 0, 0, 0.4],
    [0, 0.6, 0.4, 0],
]


def make_domain(bounds, allow_zero, max_count, n_outputs):
    return Domain.from_lists(
        inputs=[
            ContinuousInput(key=k, bounds=bounds, allow_zero=allow_zero) for k in keys
        ],
        outputs=[
            ContinuousOutput(key=f"o{i}", objective=MaximizeObjective())
            for i in range(n_outputs)
        ],
        constraints=[
            LinearEqualityConstraint(features=keys, coefficients=[1] * 4, rhs=1),
            NChooseKConstraint(
                features=keys, min_count=0, max_count=max_count, none_also_valid=False
            ),
        ],
    )


def experiments(domain):
    df = pd.DataFrame(points, columns=keys)
    for i, o in enumerate(domain.outputs.get_keys()):
        df[o] = [1.0, 2, 3, 2, 1, 4] if i == 0 else [4.0, 3, 1, 2, 3, 1]
        df[f"valid_{o}"] = 1
    return df


def run(label, strategy_cls, data_model_cls, bounds, allow_zero, max_count, n_out):
    domain = make_domain(bounds, allow_zero, max_count, n_out)
    strategy = strategy_cls(data_model=data_model_cls(domain=domain))
    strategy.tell(experiments(domain))
    try:
        x = strategy.ask(2)
        print(f"{label} bounds={bounds} allow_zero={allow_zero} max={max_count}: OK")
        print(x[keys].assign(sum=x[keys].sum(axis=1)).round(4).to_string())
    except Exception as e:
        print(
            f"{label} bounds={bounds} allow_zero={allow_zero} max={max_count}: "
            f"FAIL {type(e).__name__}: {str(e)[:200]}"
        )


for bounds, allow_zero in [((0, 0.6), False), ((0.1, 0.6), True)]:
    for max_count in (2, 3):
        run("Sobo", SoboStrategy, SoboDataModel, bounds, allow_zero, max_count, 1)
        run("Mobo", MoboStrategy, MoboDataModel, bounds, allow_zero, max_count, 2)
