import random

import pandas as pd
import pytest

from bofire.benchmarks.multi import DTLZ2
from bofire.benchmarks.single import Ackley
from bofire.samplers import PolytopeSampler
from bofire.strategies.botorch.qehvi import BoTorchQehviStrategy, BoTorchQnehviStrategy
from bofire.strategies.botorch.sobo import BoTorchSoboMultiplicativeStrategy
from tests.bofire.strategies.botorch.test_qehvi import VALID_BOTORCH_QEHVI_STRATEGY_SPEC
from tests.bofire.strategies.botorch.test_sobo import VALID_BOTORCH_SOBO_STRATEGY_SPEC

STRATEGY_SPECS_SINGLE_OBJECTIVE = {
    # BoTorchSoboAdditiveStrategy: VALID_BOTORCH_SOBO_STRATEGY_SPEC,
    BoTorchSoboMultiplicativeStrategy: VALID_BOTORCH_SOBO_STRATEGY_SPEC,
    # TODO: comment in, when BanditBO is merged in
    # BoTorchBanditBoAdditiveStrategy: VALID_BOTORCH_BANDIT_BO_STRATEGY_SPEC,
    # BoTorchBanditBoMultiplicativeStrategy: VALID_BOTORCH_BANDIT_BO_STRATEGY_SPEC
}
STRATEGY_SPECS_MULTI_OBJECTIVE = {
    BoTorchQehviStrategy: VALID_BOTORCH_QEHVI_STRATEGY_SPEC,
    BoTorchQnehviStrategy: VALID_BOTORCH_SOBO_STRATEGY_SPEC,
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
    random_strategy = PolytopeSampler(domain=benchmark.domain)
    experiments = benchmark.f(random_strategy.ask(n=10), return_complete=True)

    # set up of the strategy
    strategy = cls(**{**spec, "domain": benchmark.domain})
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
    random_strategy = PolytopeSampler(domain=benchmark.domain)
    experiments = benchmark.f(random_strategy.ask(n=10), return_complete=True)

    # set up of the strategy
    strategy = cls(
        **{**spec, "domain": benchmark.domain},
        # domain=benchmark.domain,
        ref_point=benchmark.ref_point if use_ref_point else None
    )
    strategy.tell(experiments)

    # ask
    candidates = strategy.ask(candidate_count=candidate_count)

    assert isinstance(candidates, pd.DataFrame)
    assert len(candidates) == candidate_count

    # TODO: add check of convergence towards the optimum (0)
