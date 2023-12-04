from itertools import chain

import numpy as np
import pytest
import torch
from botorch.acquisition.multi_objective import (  # qLogExpectedHypervolumeImprovement,; qLogNoisyExpectedHypervolumeImprovement,
    qExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.logei import (  # qExpectedHypervolumeImprovement,
    qLogExpectedHypervolumeImprovement,
    qLogNoisyExpectedHypervolumeImprovement,
)

# qNoisyExpectedHypervolumeImprovement,
from botorch.acquisition.multi_objective.objective import GenericMCMultiOutputObjective

import bofire.data_models.acquisition_functions.api as acquisitions
import bofire.data_models.strategies.api as data_models
import bofire.strategies.api as strategies
from bofire.benchmarks.multi import C2DTLZ2, DTLZ2
from bofire.data_models.features.api import ContinuousOutput
from bofire.data_models.strategies.api import (
    PolytopeSampler as PolytopeSamplerDataModel,
)
from bofire.strategies.api import PolytopeSampler
from tests.bofire.utils.test_multiobjective import (
    dfs,
    invalid_domains,
    valid_constrained_domains,
    valid_domains,
)


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
def test_invalid_mobo(domain, ref_point):
    with pytest.raises(ValueError):
        data_models.MoboStrategy(domain=domain, ref_point=ref_point)


@pytest.mark.parametrize("domain", valid_constrained_domains)
def test_qnehvi_valid_constrained_objectives(domain):
    data_models.MoboStrategy(domain=domain)


@pytest.mark.parametrize(
    "domain, ref_point, experiments, expected",
    [
        (valid_domains[0], {"of1": 0.5, "of2": 10.0}, dfs[0], [0.5, -10.0]),
        (valid_domains[1], {"of1": 0.5, "of3": 0.5}, dfs[1], [0.5, 0.5]),
        (valid_domains[0], None, dfs[0], [1.0, -5.0]),
        (valid_domains[1], None, dfs[1], [1.0, 2.0]),
    ],
)
def test_mobo_get_adjusted_refpoint(domain, ref_point, experiments, expected):
    data_model = data_models.MoboStrategy(domain=domain, ref_point=ref_point)
    strategy = strategies.map(data_model)
    # hack for the test to prevent training of the model when using tell
    strategy.set_experiments(experiments)
    adjusted_ref_point = strategy.get_adjusted_refpoint()
    assert isinstance(adjusted_ref_point, list)
    assert np.allclose(expected, np.asarray(adjusted_ref_point))


@pytest.mark.parametrize(
    "strategy, use_ref_point, acqf",
    [
        (data_models.MoboStrategy, use_ref_point, acqf)
        for use_ref_point in [True, False]
        for acqf in [
            acquisitions.qEHVI,
            acquisitions.qLogEHVI,
            acquisitions.qNEHVI,
            acquisitions.qLogNEHVI,
        ]
    ],
)
def test_mobo(strategy, use_ref_point, acqf):
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
        acquisition_function=acqf(),
    )
    my_strategy = strategies.map(data_model)
    my_strategy.tell(experiments)

    bacqf = my_strategy._get_acqfs(2)[0]

    assert isinstance(bacqf.objective, GenericMCMultiOutputObjective)
    if isinstance(acqf, acquisitions.qEHVI):
        assert isinstance(bacqf, qExpectedHypervolumeImprovement)
    elif isinstance(acqf, acquisitions.qNEHVI):
        assert isinstance(bacqf, qNoisyExpectedHypervolumeImprovement)
    elif isinstance(acqf, acquisitions.qLogNEHVI):
        assert isinstance(bacqf, qLogNoisyExpectedHypervolumeImprovement)
    elif isinstance(acqf, acquisitions.qLogEHVI):
        assert isinstance(bacqf, qLogExpectedHypervolumeImprovement)


@pytest.mark.parametrize(
    "acqf",
    [
        acquisitions.qEHVI,
        acquisitions.qLogEHVI,
        acquisitions.qNEHVI,
        acquisitions.qLogNEHVI,
    ],
)
def test_mobo_constraints(acqf):
    benchmark = C2DTLZ2(dim=4)
    random_strategy = PolytopeSampler(
        data_model=PolytopeSamplerDataModel(domain=benchmark.domain)
    )
    experiments = benchmark.f(random_strategy._ask(n=10), return_complete=True)
    data_model = data_models.MoboStrategy(
        domain=benchmark.domain,
        ref_point={"f_0": 1.1, "f_1": 1.1},
        acquisition_function=acqf(),
    )
    my_strategy = strategies.map(data_model)
    my_strategy.tell(experiments)
    bacqf = my_strategy._get_acqfs(2)[0]
    assert isinstance(bacqf.objective, GenericMCMultiOutputObjective)
    if isinstance(acqf, acquisitions.qEHVI):
        assert isinstance(bacqf, qExpectedHypervolumeImprovement)
    elif isinstance(acqf, acquisitions.qNEHVI):
        assert isinstance(bacqf, qNoisyExpectedHypervolumeImprovement)
    elif isinstance(acqf, acquisitions.qLogNEHVI):
        assert isinstance(bacqf, qLogNoisyExpectedHypervolumeImprovement)
    elif isinstance(acqf, acquisitions.qLogEHVI):
        assert isinstance(bacqf, qLogExpectedHypervolumeImprovement)
    assert bacqf.eta == torch.tensor(1e-3)
    assert len(bacqf.constraints) == 1
    assert torch.allclose(
        bacqf.ref_point,
        torch.tensor([-1.1, -1.1], dtype=torch.double),
    )


@pytest.mark.parametrize(
    "num_experiments, num_candidates",
    [
        (num_experiments, num_candidates)
        for num_experiments in range(8, 10)
        for num_candidates in range(1, 3)
    ],
)
@pytest.mark.slow
def test_get_acqf_input(num_experiments, num_candidates):
    # generate data
    benchmark = DTLZ2(dim=6)
    random_strategy = PolytopeSampler(
        data_model=PolytopeSamplerDataModel(domain=benchmark.domain)
    )
    experiments = benchmark.f(
        random_strategy._ask(n=num_experiments), return_complete=True
    )
    data_model = data_models.MoboStrategy(domain=benchmark.domain)
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
    data_model = data_models.MoboStrategy(
        domain=domain, ref_point={"f_0": 1.1, "f_1": 1.1}
    )
    recommender = strategies.map(data_model=data_model)
    recommender.tell(experiments=experiments)
    candidates = recommender.ask(candidate_count=1)
    recommender.to_candidates(candidates)
