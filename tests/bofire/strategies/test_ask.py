import random

import pandas as pd
import pytest

import bofire.data_models.strategies.api as data_models
import bofire.strategies.api as strategies
from bofire.benchmarks.multi import DTLZ2
from bofire.benchmarks.single import Ackley
from bofire.data_models.strategies.api import (
    PolytopeSampler as PolytopeSamplerDataModel,
)
from bofire.strategies.api import PolytopeSampler
from tests.bofire.strategies.test_qehvi import VALID_BOTORCH_QEHVI_STRATEGY_SPEC
from tests.bofire.strategies.test_sobo import VALID_BOTORCH_SOBO_STRATEGY_SPEC

STRATEGY_SPECS_SINGLE_OBJECTIVE = {
    # BoTorchSoboAdditiveStrategy: VALID_BOTORCH_SOBO_STRATEGY_SPEC,
    data_models.MultiplicativeSoboStrategy: VALID_BOTORCH_SOBO_STRATEGY_SPEC,
}
STRATEGY_SPECS_MULTI_OBJECTIVE = {
    data_models.QehviStrategy: VALID_BOTORCH_QEHVI_STRATEGY_SPEC,
    data_models.QnehviStrategy: VALID_BOTORCH_QEHVI_STRATEGY_SPEC,
}


# TODO: check this properly
@pytest.mark.parametrize(
    "cls, spec, categorical, descriptor, candidate_count",
    [
        (cls, specs, categorical, descriptor, random.randint(1, 2))
        for cls, specs in STRATEGY_SPECS_SINGLE_OBJECTIVE.items()
        for categorical in [True, False]
        for descriptor in [True, False]
    ],
)
@pytest.mark.slow
def test_ask_single_objective(cls, spec, categorical, descriptor, candidate_count):
    # generate data
    benchmark = Ackley(categorical=categorical, descriptor=descriptor)
    random_strategy = PolytopeSampler(
        data_model=PolytopeSamplerDataModel(domain=benchmark.domain)
    )
    experiments = benchmark.f(random_strategy.ask(10), return_complete=True)

    # set up of the strategy
    data_model = cls(**{**spec, "domain": benchmark.domain})
    print("data_model:", type(data_model))
    strategy = strategies.map(data_model)
    strategy.tell(experiments)

    # ask
    candidates = strategy.ask(candidate_count=candidate_count)

    assert isinstance(candidates, pd.DataFrame)
    assert len(candidates) == candidate_count

    # TODO: add check of convergence towards the optimum (0)


@pytest.mark.parametrize(
    "cls, spec, use_ref_point, candidate_count",
    [
        (cls, specs, use_ref_point, random.randint(1, 2))
        for cls, specs in STRATEGY_SPECS_MULTI_OBJECTIVE.items()
        for use_ref_point in [True, False]
        # for categorical in [True, False]
        # for descriptor in [True, False]
    ],
)
@pytest.mark.slow  # use pytest . --runslow in command line to include these tests
def test_ask_multi_objective(cls, spec, use_ref_point, candidate_count):
    # generate data
    benchmark = DTLZ2(
        dim=6
    )  # TODO: expand benchmark also towards categorical features?
    random_strategy = PolytopeSampler(
        data_model=PolytopeSamplerDataModel(domain=benchmark.domain)
    )
    experiments = benchmark.f(random_strategy.ask(10), return_complete=True)

    # set up of the strategy
    data_model = cls(
        **{**spec, "domain": benchmark.domain},
        # domain=benchmark.domain,
        ref_point=benchmark.ref_point if use_ref_point else None
    )
    strategy = strategies.map(data_model=data_model)
    strategy.tell(experiments)

    # ask
    candidates = strategy.ask(candidate_count=candidate_count)

    assert isinstance(candidates, pd.DataFrame)
    assert len(candidates) == candidate_count

    # TODO: add check of convergence towards the optimum (0)
