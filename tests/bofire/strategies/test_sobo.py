import math
from itertools import chain

import pytest
import torch
from botorch.acquisition import (
    qExpectedImprovement,
    qNoisyExpectedImprovement,
    qProbabilityOfImprovement,
    qSimpleRegret,
    qUpperConfidenceBound,
)
from botorch.acquisition.objective import GenericMCObjective

import bofire.data_models.strategies.api as data_models
import tests.bofire.data_models.specs.api as specs
from bofire.benchmarks.multi import DTLZ2
from bofire.benchmarks.single import Himmelblau, _CategoricalDiscreteHimmelblau
from bofire.data_models.acquisition_functions.api import qEI, qNEI, qPI, qSR, qUCB
from bofire.data_models.strategies.api import (
    PolytopeSampler as PolytopeSamplerDataModel,
)
from bofire.strategies.api import CustomSoboStrategy, PolytopeSampler, SoboStrategy
from tests.bofire.strategies.test_base import domains

# from tests.bofire.strategies.botorch.test_model_spec import VALID_MODEL_SPEC_LIST

VALID_BOTORCH_SOBO_STRATEGY_SPEC = {
    "domain": domains[1],
    "acquisition_function": specs.acquisition_functions.valid().obj(),
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
        # {**VALID_BOTORCH_SOBO_STRATEGY_SPEC, "surrogate_specs": VALID_MODEL_SPEC_LIST},
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
    data_model = data_models.SoboStrategy(domain=domain, acquisition_function=acqf)
    strategy = SoboStrategy(data_model=data_model)

    msg = "Model not trained."
    with pytest.raises(AssertionError, match=msg):
        strategy._get_acqfs(2)


@pytest.mark.parametrize(
    "acqf, expected, num_test_candidates",
    [
        (acqf_inp[0], acqf_inp[1], num_test_candidates)
        for acqf_inp in [
            (qEI(), qExpectedImprovement),
            (qNEI(), qNoisyExpectedImprovement),
            (qPI(), qProbabilityOfImprovement),
            (qUCB(), qUpperConfidenceBound),
            (qSR(), qSimpleRegret),
        ]
        for num_test_candidates in range(1, 3)
    ],
)
def test_SOBO_get_acqf(acqf, expected, num_test_candidates):
    # generate data
    benchmark = Himmelblau()

    random_strategy = PolytopeSampler(
        data_model=PolytopeSamplerDataModel(domain=benchmark.domain)
    )

    experiments = benchmark.f(random_strategy.ask(20), return_complete=True)

    data_model = data_models.SoboStrategy(
        domain=benchmark.domain, acquisition_function=acqf
    )
    strategy = SoboStrategy(data_model=data_model)

    strategy.tell(experiments)

    acqfs = strategy._get_acqfs(2)
    assert len(acqfs) == 1

    assert isinstance(acqfs[0], expected)


def test_SOBO_calc_acquisition():
    benchmark = Himmelblau()
    experiments = benchmark.f(benchmark.domain.inputs.sample(10), return_complete=True)
    samples = benchmark.domain.inputs.sample(2)
    data_model = data_models.SoboStrategy(
        domain=benchmark.domain, acquisition_function=qEI()
    )
    strategy = SoboStrategy(data_model=data_model)
    strategy.tell(experiments=experiments)
    vals = strategy.calc_acquisition(samples)
    assert len(vals) == 2
    vals = strategy.calc_acquisition(samples, combined=True)
    assert len(vals) == 1


def test_SOBO_init_qUCB():
    beta = 0.5
    acqf = qUCB(beta=beta)

    # generate data
    benchmark = Himmelblau()
    random_strategy = PolytopeSampler(
        data_model=PolytopeSamplerDataModel(domain=benchmark.domain)
    )
    experiments = benchmark.f(random_strategy._ask(n=20), return_complete=True)

    data_model = data_models.SoboStrategy(
        domain=benchmark.domain, acquisition_function=acqf
    )
    strategy = SoboStrategy(data_model=data_model)
    strategy.tell(experiments)

    acqf = strategy._get_acqfs(2)[0]
    assert acqf.beta_prime == math.sqrt(beta * math.pi / 2)


@pytest.mark.parametrize(
    "acqf, num_experiments, num_candidates",
    [
        (acqf.obj(), num_experiments, num_candidates)
        for acqf in specs.acquisition_functions.valids
        for num_experiments in range(8, 10)
        for num_candidates in range(1, 3)
    ],
)
@pytest.mark.slow
def test_get_acqf_input(acqf, num_experiments, num_candidates):
    # generate data
    benchmark = Himmelblau()
    random_strategy = PolytopeSampler(
        data_model=PolytopeSamplerDataModel(domain=benchmark.domain)
    )
    experiments = benchmark.f(
        random_strategy._ask(n=num_experiments), return_complete=True
    )

    data_model = data_models.SoboStrategy(
        domain=benchmark.domain, acquisition_function=acqf
    )
    strategy = SoboStrategy(data_model=data_model)

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
    assert X_pending.shape == (  # type: ignore
        num_candidates,
        len(set(chain(*names.values()))),
    )


def test_custom_get_objective():
    def f(samples, callables, weights, X):
        outputs_list = []
        for c, w in zip(callables, weights):
            outputs_list.append(c(samples, None) ** w)
        samples = torch.stack(outputs_list, dim=-1)

        return (samples[..., 0] + samples[..., 1]) * (samples[..., 0] * samples[..., 1])

    benchmark = DTLZ2(3)
    data_model = data_models.CustomSoboStrategy(
        domain=benchmark.domain, acquisition_function=qNEI()
    )
    strategy = CustomSoboStrategy(data_model=data_model)
    strategy.f = f
    generic_objective = strategy._get_objective()
    assert isinstance(generic_objective, GenericMCObjective)


def test_custom_get_objective_invalid():
    benchmark = DTLZ2(3)
    data_model = data_models.CustomSoboStrategy(
        domain=benchmark.domain, acquisition_function=qNEI()
    )
    strategy = CustomSoboStrategy(data_model=data_model)

    with pytest.raises(ValueError):
        strategy._get_objective()


def test_custom_dumps_loads():
    def f(samples, callables, weights, X):
        outputs_list = []
        for c, w in zip(callables, weights):
            outputs_list.append(c(samples, None) ** w)
        samples = torch.stack(outputs_list, dim=-1)

        return (samples[..., 0] + samples[..., 1]) * (samples[..., 0] * samples[..., 1])

    benchmark = DTLZ2(3)
    data_model1 = data_models.CustomSoboStrategy(
        domain=benchmark.domain,
        acquisition_function=qNEI(),
        use_output_constraints=False,
    )
    strategy1 = CustomSoboStrategy(data_model=data_model1)
    strategy1.f = f
    f_str = strategy1.dumps()

    data_model2 = data_models.CustomSoboStrategy(
        domain=benchmark.domain,
        acquisition_function=qNEI(),
        use_output_constraints=False,
        dump=f_str,
    )
    strategy2 = CustomSoboStrategy(data_model=data_model2)

    data_model3 = data_models.CustomSoboStrategy(
        domain=benchmark.domain, acquisition_function=qNEI()
    )
    strategy3 = CustomSoboStrategy(data_model=data_model3)
    strategy3.loads(f_str)

    assert isinstance(strategy2.f, type(f))
    assert isinstance(strategy3.f, type(f))

    samples = torch.rand(30, 2, requires_grad=True) * 5
    objective1 = strategy1._get_objective()
    output1 = objective1.forward(samples)
    objective2 = strategy2._get_objective()
    output2 = objective2.forward(samples)
    objective3 = strategy3._get_objective()
    output3 = objective3.forward(samples)

    torch.testing.assert_close(output1, output2)
    torch.testing.assert_close(output1, output3)


def test_custom_dumps_invalid():
    benchmark = DTLZ2(3)
    data_model = data_models.CustomSoboStrategy(
        domain=benchmark.domain, acquisition_function=qNEI()
    )
    strategy = CustomSoboStrategy(data_model=data_model)
    with pytest.raises(ValueError):
        strategy.dumps()


@pytest.mark.parametrize("candidate_count", [1, 2])
def test_sobo_fully_combinatorical(candidate_count):
    benchmark = _CategoricalDiscreteHimmelblau()

    strategy_data = data_models.SoboStrategy(domain=benchmark.domain)
    strategy = SoboStrategy(data_model=strategy_data)

    experiments = benchmark.f(benchmark.domain.inputs.sample(10), return_complete=True)

    strategy.tell(experiments=experiments)
    strategy.ask(candidate_count=candidate_count)
