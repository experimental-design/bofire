import math
import random

import pytest
import torch
from botorch.acquisition.monte_carlo import (
    qExpectedImprovement,
    qNoisyExpectedImprovement,
    qProbabilityOfImprovement,
    qSimpleRegret,
    qUpperConfidenceBound,
)

from bofire.strategies.botorch.sobo import (
    BoTorchSoboStrategy,
    qEI,
    qNEI,
    qPI,
    qSR,
    qUCB,
)
from bofire.utils.enum import AcquisitionFunctionEnum
from tests.bofire.domain.test_domain_validators import generate_experiments
from tests.bofire.strategies.botorch.test_base import domains

# from tests.bofire.strategies.botorch.test_model_spec import VALID_MODEL_SPEC_LIST

VALID_BOTORCH_SOBO_STRATEGY_SPEC = {
    "domain": domains[2],
    "acquisition_function": random.choice(list(AcquisitionFunctionEnum)),
    # "num_sobol_samples": 1024,
    # "num_restarts": 8,
    # "num_raw_samples": 1024,
    "descriptor_method": "EXHAUSTIVE",
    "categorical_method": "EXHAUSTIVE",
}

BOTORCH_SOBO_STRATEGY_SPECS = {
    "valids": [
        VALID_BOTORCH_SOBO_STRATEGY_SPEC,
        {**VALID_BOTORCH_SOBO_STRATEGY_SPEC, "seed": 1},
        # {**VALID_BOTORCH_SOBO_STRATEGY_SPEC, "model_specs": VALID_MODEL_SPEC_LIST},
    ],
    "invalids": [
        {**VALID_BOTORCH_SOBO_STRATEGY_SPEC, "acquisition_function": None},
        {**VALID_BOTORCH_SOBO_STRATEGY_SPEC, "descriptor_method": None},
        {**VALID_BOTORCH_SOBO_STRATEGY_SPEC, "categorical_method": None},
        {**VALID_BOTORCH_SOBO_STRATEGY_SPEC, "seed": -1},
    ],
}


@pytest.mark.parametrize(
    "domain, acqf",
    [(domains[0], VALID_BOTORCH_SOBO_STRATEGY_SPEC["acquisition_function"])],
)
def test_SOBO_not_fitted(domain, acqf):
    strategy = BoTorchSoboStrategy(domain=domain, acquisition_function=acqf)
    msg = "Model not trained."
    with pytest.raises(AssertionError, match=msg):
        strategy._init_acqf()


@pytest.mark.parametrize(
    "domain, acqf, expected, num_test_candidates",
    [
        (domains[0], acqf_inp[0], acqf_inp[1], num_test_candidates)
        for acqf_inp in [
            ("QEI", qExpectedImprovement),
            ("QNEI", qNoisyExpectedImprovement),
            ("QPI", qProbabilityOfImprovement),
            ("QUCB", qUpperConfidenceBound),
            ("QSR", qSimpleRegret),
            (qEI(), qExpectedImprovement),
            (qNEI(), qNoisyExpectedImprovement),
            (qPI(), qProbabilityOfImprovement),
            (qUCB(), qUpperConfidenceBound),
            (qSR(), qSimpleRegret),
        ]
        for num_test_candidates in range(1, 3)
    ],
)
def test_SOBO_init_acqf(domain, acqf, expected, num_test_candidates):
    strategy = BoTorchSoboStrategy(domain=domain, acquisition_function=acqf)
    experiments = generate_experiments(domain, 20)
    experiments_test = generate_experiments(domain, num_test_candidates)

    strategy.tell(experiments)
    assert isinstance(strategy.acqf, expected)
    # test acqf calc
    acqf_vals = strategy._choose_from_pool(experiments_test, num_test_candidates)
    assert acqf_vals.shape[0] == num_test_candidates
    domain.experiments = None


def test_SOBO_init_qUCB():
    beta = 0.5
    acqf = qUCB(beta=beta)
    strategy = BoTorchSoboStrategy(domain=domains[0], acquisition_function=acqf)
    experiments = generate_experiments(domains[0], 20)
    strategy.tell(experiments)
    assert strategy.acqf.beta_prime == math.sqrt(beta * math.pi / 2)
    domains[0].experiments = None


@pytest.mark.parametrize(
    "domain, acqf, num_experiments, num_candidates",
    [
        (domains[0], acqf, num_experiments, num_candidates)
        for acqf in ["QEI", "QNEI", "QPI", "QUCB", "QSR"]
        for num_experiments in range(8, 10)
        for num_candidates in range(1, 3)
    ],
)
def test_get_acqf_input(domain, acqf, num_experiments, num_candidates):

    strategy = BoTorchSoboStrategy(domain=domain, acquisition_function=acqf)

    experiments = generate_experiments(domain, num_experiments)

    # just to ensure there are no former experiments/ candidates already stored in the domain
    strategy.domain.experiments = None
    strategy.domain.candidates = None

    strategy.tell(experiments)
    strategy.ask(candidate_count=num_candidates, add_pending=True)

    X_train, X_pending = strategy.get_acqf_input_tensors()

    assert torch.is_tensor(X_train)
    assert torch.is_tensor(X_pending)
    assert X_train.shape == (num_experiments, len(domain.input_features.get_keys()))
    assert X_pending.shape == (num_candidates, len(domain.input_features.get_keys()))
