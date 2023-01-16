import random
from itertools import chain

import numpy as np
import pytest
import torch
from botorch.acquisition.multi_objective import (
    qExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.objective import WeightedMCMultiOutputObjective

from bofire.benchmarks.multi import DTLZ2
from bofire.domain.domain import Domain
from bofire.domain.features import ContinuousInput, ContinuousOutput
from bofire.domain.objectives import MaximizeObjective, MinimizeObjective
from bofire.samplers import PolytopeSampler
from bofire.strategies.botorch.qehvi import BoTorchQehviStrategy, BoTorchQnehviStrategy
from bofire.utils.enum import CategoricalMethodEnum
from tests.bofire.domain.test_features import VALID_CONTINUOUS_INPUT_FEATURE_SPEC
from tests.bofire.utils.test_multiobjective import dfs, invalid_domains, valid_domains

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
    "domain": Domain(input_features=[if1, if2, if3], output_features=[of1, of2]),
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
        # {**VALID_BOTORCH_QEHVI_STRATEGY_SPEC, "model_specs": VALID_MODEL_SPEC_LIST},
        {**VALID_BOTORCH_QEHVI_STRATEGY_SPEC, "ref_point": {"of1": 1.0, "of2": 2}},
    ],
    "invalids": [
        {**VALID_BOTORCH_QEHVI_STRATEGY_SPEC, "descriptor_method": None},
        {**VALID_BOTORCH_QEHVI_STRATEGY_SPEC, "categorical_method": None},
        {**VALID_BOTORCH_QEHVI_STRATEGY_SPEC, "seed": -1},
    ],
}


@pytest.mark.parametrize(
    "domain, ref_point",
    [(invalid_domains[0], None), (invalid_domains[1], None), (valid_domains[0], [0])],
)
def test_invalid_qehvi_init_domain(domain, ref_point):
    with pytest.raises(ValueError):
        BoTorchQehviStrategy(domain=domain, ref_point=ref_point)


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
    strategy = BoTorchQehviStrategy(domain=domain, ref_point=ref_point)
    # hack for the test to prevent training of the model when using tell
    strategy.domain.set_experiments(experiments)
    adjusted_ref_point = strategy.get_adjusted_refpoint()
    assert isinstance(adjusted_ref_point, list)
    assert np.allclose(expected, np.asarray(adjusted_ref_point))


@pytest.mark.parametrize(
    "strategy, use_ref_point, num_test_candidates",
    [
        (strategy, use_ref_point, num_test_candidates)
        for strategy in [BoTorchQehviStrategy, BoTorchQnehviStrategy]
        for use_ref_point in [True, False]
        for num_test_candidates in range(1, 3)
    ],
)
def test_qehvi(strategy, use_ref_point, num_test_candidates):
    # generate data
    benchmark = DTLZ2(dim=6)
    random_strategy = PolytopeSampler(domain=benchmark.domain)
    experiments = benchmark.f(random_strategy._sample(n=10), return_complete=True)
    experiments_test = benchmark.f(
        random_strategy._sample(n=num_test_candidates), return_complete=True
    )
    # init strategy
    my_strategy = strategy(
        domain=benchmark.domain,
        ref_point=benchmark.ref_point if use_ref_point else None,
    )
    my_strategy.tell(experiments)
    assert isinstance(my_strategy.objective, WeightedMCMultiOutputObjective)
    assert isinstance(
        my_strategy.acqf,
        qExpectedHypervolumeImprovement
        if strategy == BoTorchQehviStrategy
        else qNoisyExpectedHypervolumeImprovement,
    )
    # test acqf calc
    acqf_vals = my_strategy._choose_from_pool(experiments_test, num_test_candidates)
    assert acqf_vals.shape[0] == num_test_candidates


@pytest.mark.parametrize(
    "strategy, num_experiments, num_candidates",
    [
        (strategy, num_experiments, num_candidates)
        for strategy in [BoTorchQehviStrategy, BoTorchQnehviStrategy]
        for num_experiments in range(8, 10)
        for num_candidates in range(1, 3)
    ],
)
@pytest.mark.slow
def test_get_acqf_input(strategy, num_experiments, num_candidates):

    strategy = strategy(**VALID_BOTORCH_QEHVI_STRATEGY_SPEC)

    # generate data
    benchmark = DTLZ2(dim=6)
    random_strategy = PolytopeSampler(domain=benchmark.domain)
    experiments = benchmark.f(
        random_strategy._sample(n=num_experiments), return_complete=True
    )

    # just to ensure there are no former experiments/ candidates already stored in the domain
    strategy.domain.experiments = None
    strategy.domain.candidates = None

    strategy.tell(experiments)
    strategy.ask(candidate_count=num_candidates, add_pending=True)

    X_train, X_pending = strategy.get_acqf_input_tensors()

    _, names = strategy.domain.input_features._get_transform_info(
        specs=strategy.model_specs.input_preprocessing_specs
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
