import random

import pandas as pd
import pytest

import bofire.data_models.strategies.api as data_models
import bofire.strategies.api as strategies
from bofire.benchmarks.multi import DTLZ2
from bofire.benchmarks.single import Ackley
from bofire.data_models.strategies.api import RandomStrategy as RandomStrategyDataModel
from bofire.strategies.api import RandomStrategy
from tests.bofire.strategies.test_qehvi import VALID_BOTORCH_QEHVI_STRATEGY_SPEC
from tests.bofire.strategies.test_sobo import VALID_BOTORCH_SOBO_STRATEGY_SPEC


STRATEGY_SPECS_SINGLE_OBJECTIVE = {
    data_models.SoboStrategy: VALID_BOTORCH_SOBO_STRATEGY_SPEC,
}


STRATEGY_SPECS_MULTI_OBJECTIVE = {
    data_models.QehviStrategy: VALID_BOTORCH_QEHVI_STRATEGY_SPEC,
    data_models.QnehviStrategy: VALID_BOTORCH_QEHVI_STRATEGY_SPEC,
    data_models.AdditiveSoboStrategy: VALID_BOTORCH_QEHVI_STRATEGY_SPEC,
    data_models.MultiplicativeSoboStrategy: VALID_BOTORCH_QEHVI_STRATEGY_SPEC,
    data_models.MultiplicativeAdditiveSoboStrategy: VALID_BOTORCH_QEHVI_STRATEGY_SPEC,
}
mo_strategy_has_ref_point = {
    data_models.QehviStrategy: True,
    data_models.QnehviStrategy: True,
    data_models.AdditiveSoboStrategy: False,
    data_models.MultiplicativeSoboStrategy: False,
    data_models.MultiplicativeAdditiveSoboStrategy: False,
}
mo_strategy_has_additive_objective = {
    data_models.QehviStrategy: False,
    data_models.QnehviStrategy: False,
    data_models.AdditiveSoboStrategy: False,
    data_models.MultiplicativeSoboStrategy: False,
    data_models.MultiplicativeAdditiveSoboStrategy: True,
}
mo_strategy_support_weights = {
    data_models.QehviStrategy: False,
    data_models.QnehviStrategy: False,
    data_models.AdditiveSoboStrategy: True,
    data_models.MultiplicativeSoboStrategy: True,
    data_models.MultiplicativeAdditiveSoboStrategy: True,
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
# @pytest.mark.slow
def test_ask_single_objective(cls, spec, categorical, descriptor, candidate_count):
    # generate data
    benchmark = Ackley(categorical=categorical, descriptor=descriptor)
    random_strategy = RandomStrategy(
        data_model=RandomStrategyDataModel(domain=benchmark.domain),
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
    "cls, spec, use_ref_point, add_additive_features, vary_weights, candidate_count",
    [
        (
            cls,
            specs,
            use_ref_point,
            add_additive_features,
            vary_weights,
            random.randint(1, 2),
        )
        for cls, specs in STRATEGY_SPECS_MULTI_OBJECTIVE.items()
        for use_ref_point in [True, False]
        for add_additive_features in [True, False]
        for vary_weights in [True, False]
        # for categorical in [True, False]
        # for descriptor in [True, False]
    ],
)
# @pytest.mark.slow  # use pytest . --runslow in command line to include these tests
def test_ask_multi_objective(
    cls, spec, use_ref_point, add_additive_features, vary_weights, candidate_count
):
    # generate data
    benchmark = DTLZ2(
        dim=6,
    )  # TODO: expand benchmark also towards categorical features?
    random_strategy = RandomStrategy(
        data_model=RandomStrategyDataModel(domain=benchmark.domain),
    )
    experiments = benchmark.f(random_strategy.ask(10), return_complete=True)

    if vary_weights:
        if not mo_strategy_support_weights[cls]:
            print(f"Skip test: Weights not supported by {cls}.")
            return
        obj = benchmark.domain.outputs.get_by_objective()
        obj[-1].objective.w = 0.5

    kwargs = {**spec, "domain": benchmark.domain}
    # skip tests, if kw-args (ref_point, additive_features) are not supported by the strategy
    if use_ref_point:
        if not mo_strategy_has_ref_point[cls]:
            print(f"Skip test: Ref point not supported by {cls}.")
            return
        kwargs["ref_point"] = benchmark.ref_point
    if add_additive_features:
        if not mo_strategy_has_additive_objective[cls]:
            print(f"Skip test: Additive features not supported by {cls}.")
            return
        kwargs["additive_features"] = [benchmark.domain.outputs.get_keys()[0]]

    # set up of the strategy
    data_model = cls(**kwargs)

    strategy = strategies.map(data_model=data_model)
    strategy.tell(experiments)

    # ask
    candidates = strategy.ask(candidate_count=candidate_count)

    assert isinstance(candidates, pd.DataFrame)
    assert len(candidates) == candidate_count

    # TODO: add check of convergence towards the optimum (0)
