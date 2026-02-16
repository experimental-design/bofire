from bofire.benchmarks.api import MOMFBraninCurrin
from bofire.data_models.acquisition_functions.api import qMFHVKG
from bofire.data_models.enum import SamplingMethodEnum
from bofire.data_models.features.api import ContinuousTaskInput
from bofire.data_models.strategies.api import BotorchOptimizer
from bofire.strategies.api import MultiFidelityHVKGStrategy, RandomStrategy


def test_mfhvkg_fidelity_selection():
    benchmark = MOMFBraninCurrin()

    random_strategy = RandomStrategy.make(
        domain=benchmark.domain,
        fallback_sampling_method=SamplingMethodEnum.SOBOL,
        seed=42,
    )

    experiments = benchmark.f(random_strategy.ask(4), return_complete=True)

    # we reduce the samples to improve runtime
    strategy = MultiFidelityHVKGStrategy.make(
        domain=benchmark.domain,
        acquisition_function=qMFHVKG(n_mc_samples=16),
        acquisition_optimizer=BotorchOptimizer(
            n_raw_samples=128,
            maxiter=200,
        ),
        seed=0,
    )

    strategy.tell(experiments)
    preds = strategy.predict(experiments)
    assert len(preds) == len(experiments)

    candidate_count = 3
    candidates = strategy.ask(candidate_count)
    assert len(candidates) == candidate_count

    # when the high fidelity is expensive, MFHVKG will (almost always) evaluate the low fidelity
    task_input: ContinuousTaskInput = strategy.domain.inputs.get(ContinuousTaskInput)[0]
    task_input.fidelity_cost.weight = 10.0
    candidate_lf = strategy.ask(1)
    assert candidate_lf[task_input.key].item() == task_input.lower_bound

    # when the high fidelity is cheap, MFHVKG will evaluate a higher fidelity
    task_input: ContinuousTaskInput = strategy.domain.inputs.get(ContinuousTaskInput)[0]
    task_input.fidelity_cost.weight = 0.01
    candidate_lf = strategy.ask(1)
    assert candidate_lf[task_input.key].item() > 0.1


# TODO: write test for CategoricalTaskInput
