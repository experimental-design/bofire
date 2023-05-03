import random
from itertools import chain

import numpy as np
import pytest
import torch
from botorch.acquisition.multi_objective import (
    qExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.objective import GenericMCMultiOutputObjective

import bofire.data_models.strategies.api as data_models
import bofire.strategies.api as strategies
from bofire.benchmarks.multi import C2DTLZ2, DTLZ2
from bofire.data_models.domain.api import Domain
from bofire.data_models.enum import CategoricalMethodEnum
from bofire.data_models.features.api import ContinuousInput, ContinuousOutput
from bofire.data_models.objectives.api import MaximizeObjective, MinimizeObjective
from bofire.data_models.strategies.api import (
    PolytopeSampler as PolytopeSamplerDataModel,
)
from bofire.strategies.api import PolytopeSampler
from tests.bofire.strategies.specs import VALID_CONTINUOUS_INPUT_FEATURE_SPEC
from tests.bofire.utils.test_multiobjective import (
    dfs,
    invalid_domains,
    valid_constrained_domains,
    valid_domains,
)

if1 = ContinuousInput(
    **{
        **VALID_CONTINUOUS_INPUT_FEATURE_SPEC,
        "key": "if1",
    }
)

if2 = ContinuousInput(
    **{
        **VALID_CONTINUOUS_INPUT_FEATURE_SPEC,
        "key": "if2",
    }
)

if3 = ContinuousInput(
    **{
        **VALID_CONTINUOUS_INPUT_FEATURE_SPEC,
        "key": "if3",
    }
)

of1 = ContinuousOutput(
    objective=MaximizeObjective(w=1),
    key="of1",
)

of2 = ContinuousOutput(
    objective=MinimizeObjective(w=1),
    key="of2",
)

VALID_BOTORCH_QEHVI_STRATEGY_SPEC = {
    "domain": Domain.from_lists(inputs=[if1, if2, if3], outputs=[of1, of2]),
    # "num_sobol_samples": 1024,
    # "num_restarts": 8,
    # "num_raw_samples": 1024,
    "descriptor_method": random.choice(list(CategoricalMethodEnum)),
    "categorical_method": "EXHAUSTIVE",
}

BOTORCH_QEHVI_STRATEGY_SPECS = {
    "valids": [
        VALID_BOTORCH_QEHVI_STRATEGY_SPEC,
        {**VALID_BOTORCH_QEHVI_STRATEGY_SPEC, "seed": 1},
        # {**VALID_BOTORCH_QEHVI_STRATEGY_SPEC, "surrogate_specs": VALID_MODEL_SPEC_LIST},
        {**VALID_BOTORCH_QEHVI_STRATEGY_SPEC, "ref_point": {"of1": 1.0, "of2": 2}},
    ],
    "invalids": [
        {**VALID_BOTORCH_QEHVI_STRATEGY_SPEC, "descriptor_method": None},
        {**VALID_BOTORCH_QEHVI_STRATEGY_SPEC, "categorical_method": None},
        # {**VALID_BOTORCH_QEHVI_STRATEGY_SPEC, "seed": -1},
    ],
}


@pytest.mark.parametrize(
    "domain, ref_point",
    [
        (invalid_domains[0], None),
        (invalid_domains[1], None),
        (valid_domains[0], [0]),
        (valid_domains[0], {}),
        (valid_domains[0], {"of1": 0.0, "of2": 0, "of3": 0}),
        (valid_domains[0], {"of1": 0.0}),
        (valid_domains[0], {"of1": 0.0, "of3": 0.0}),
    ],
)
def test_invalid_qehvi(domain, ref_point):
    with pytest.raises(ValueError):
        data_models.QehviStrategy(domain=domain, ref_point=ref_point)


@pytest.mark.parametrize("domain", valid_constrained_domains)
def test_qehvi_invalid_constrained_objectives(domain):
    with pytest.raises(ValueError):
        data_models.QehviStrategy(domain=domain)


@pytest.mark.parametrize("domain", valid_constrained_domains)
def test_qnehvi_valid_constrained_objectives(domain):
    data_models.QnehviStrategy(domain=domain)


@pytest.mark.parametrize(
    "domain, ref_point, experiments, expected",
    [
        (valid_domains[0], {"of1": 0.5, "of2": 10.0}, dfs[0], [0.5, -10.0]),
        (valid_domains[1], {"of1": 0.5, "of3": 0.5}, dfs[1], [0.5, 0.5]),
        (valid_domains[0], None, dfs[0], [1.0, -5.0]),
        (valid_domains[1], None, dfs[1], [1.0, 2.0]),
    ],
)
def test_qehvi_get_adjusted_refpoint(domain, ref_point, experiments, expected):
    data_model = data_models.QehviStrategy(domain=domain, ref_point=ref_point)
    strategy = strategies.map(data_model)
    # hack for the test to prevent training of the model when using tell
    strategy.set_experiments(experiments)
    adjusted_ref_point = strategy.get_adjusted_refpoint()
    assert isinstance(adjusted_ref_point, list)
    assert np.allclose(expected, np.asarray(adjusted_ref_point))


@pytest.mark.parametrize(
    "strategy, use_ref_point, num_test_candidates",
    [
        (strategy, use_ref_point, num_test_candidates)
        for strategy in [data_models.QehviStrategy, data_models.QnehviStrategy]
        for use_ref_point in [True, False]
        for num_test_candidates in range(1, 3)
    ],
)
def test_qehvi(strategy, use_ref_point, num_test_candidates):
    # generate data
    benchmark = DTLZ2(dim=6)
    random_strategy = PolytopeSampler(
        data_model=PolytopeSamplerDataModel(domain=benchmark.domain)
    )
    experiments = benchmark.f(random_strategy._ask(n=10), return_complete=True)
    # init strategy
    data_model = strategy(
        domain=benchmark.domain,
        ref_point=benchmark.ref_point if use_ref_point else None,
    )
    my_strategy = strategies.map(data_model)
    my_strategy.tell(experiments)

    acqf = my_strategy._get_acqfs(2)[0]

    assert isinstance(acqf.objective, GenericMCMultiOutputObjective)
    assert isinstance(
        acqf,
        qExpectedHypervolumeImprovement
        if strategy == data_models.QehviStrategy
        else qNoisyExpectedHypervolumeImprovement,
    )
    # test acqf calc
    # acqf_vals = my_strategy._choose_from_pool(experiments_test, num_test_candidates)
    # assert acqf_vals.shape[0] == num_test_candidates


def test_qnehvi_constraints():
    benchmark = C2DTLZ2(dim=4)
    random_strategy = PolytopeSampler(
        data_model=PolytopeSamplerDataModel(domain=benchmark.domain)
    )
    experiments = benchmark.f(random_strategy._ask(n=10), return_complete=True)
    data_model = data_models.QnehviStrategy(
        domain=benchmark.domain, ref_point={"f_0": 1.1, "f_1": 1.1}
    )
    my_strategy = strategies.map(data_model)
    my_strategy.tell(experiments)
    acqf = my_strategy._get_acqfs(2)[0]
    assert isinstance(acqf.objective, GenericMCMultiOutputObjective)
    assert isinstance(acqf, qNoisyExpectedHypervolumeImprovement)
    assert acqf.eta == torch.tensor(1e-3)
    assert len(acqf.constraints) == 1
    assert torch.allclose(
        acqf.ref_point,
        torch.tensor([-1.1, -1.1], dtype=torch.double),
    )


@pytest.mark.parametrize(
    "strategy, ref_point, num_experiments, num_candidates",
    [
        (strategy, ref_point, num_experiments, num_candidates)
        for strategy in [data_models.QehviStrategy, data_models.QnehviStrategy]
        for ref_point in [None, {"f_0": 1.1, "f_1": 1.1}]
        for num_experiments in range(8, 10)
        for num_candidates in range(1, 3)
    ],
)
@pytest.mark.slow
def test_get_acqf_input(strategy, ref_point, num_experiments, num_candidates):
    # generate data
    benchmark = DTLZ2(dim=6)
    random_strategy = PolytopeSampler(
        data_model=PolytopeSamplerDataModel(domain=benchmark.domain)
    )
    experiments = benchmark.f(
        random_strategy._ask(n=num_experiments), return_complete=True
    )
    data_model = strategy(domain=benchmark.domain)
    strategy = strategies.map(data_model)
    # , ref_point=ref_pointw

    strategy.tell(experiments)
    strategy.ask(candidate_count=num_candidates, add_pending=True)

    X_train, X_pending = strategy.get_acqf_input_tensors()

    _, names = strategy.domain.inputs._get_transform_info(
        specs=strategy.surrogate_specs.input_preprocessing_specs
    )

    assert torch.is_tensor(X_train)
    assert torch.is_tensor(X_pending)
    assert X_train.shape == (
        num_experiments,
        len(set(chain(*names.values()))),
    )
    assert X_pending.shape == (
        num_candidates,
        len(set(chain(*names.values()))),
    )


def test_no_objective():
    domain = DTLZ2(dim=6).domain
    experiments = DTLZ2(dim=6).f(domain.inputs.sample(10), return_complete=True)
    domain.outputs.features.append(ContinuousOutput(key="ignore", objective=None))
    experiments["ignore"] = experiments["f_0"] + 6
    experiments["valid_ignore"] = 1
    data_model = data_models.QehviStrategy(
        domain=domain, ref_point={"f_0": 1.1, "f_1": 1.1}
    )
    recommender = strategies.map(data_model=data_model)
    recommender.tell(experiments=experiments)
    candidates = recommender.ask(candidate_count=1)
    recommender.to_candidates(candidates)
