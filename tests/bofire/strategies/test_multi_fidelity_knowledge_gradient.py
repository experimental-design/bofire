from bofire.benchmarks.api import MOMFBraninCurrin
from bofire.data_models.acquisition_functions.api import qMFHVKG
from bofire.data_models.enum import SamplingMethodEnum
from bofire.data_models.features.api import ContinuousTaskInput
from bofire.data_models.strategies.api import BotorchOptimizer
from bofire.data_models.surrogates.api import LinearDeterministicSurrogate
from bofire.strategies.api import MultiFidelityHVKGStrategy, RandomStrategy


def test_mfhvkg_fidelity_selection():
    # FIXME: this currently fails: botorch is struggling with the reference point
    # having 2 outputs, but the model producing 3 outputs.
    benchmark = MOMFBraninCurrin()

    random_strategy = RandomStrategy.make(
        domain=benchmark.domain,
        fallback_sampling_method=SamplingMethodEnum.SOBOL,
        seed=42,
    )

    experiments = benchmark.f(random_strategy.ask(4), return_complete=True)

    strategy = MultiFidelityHVKGStrategy.make(
        domain=benchmark.domain,
        acquisition_function=qMFHVKG(n_mc_samples=16),
        acquisition_optimizer=BotorchOptimizer(
            n_raw_samples=128,
            maxiter=200,
        ),
        seed=0,
        fidelity_cost_output_key="fidelity_cost",
    )

    strategy.tell(experiments)
    preds = strategy.predict(experiments)
    assert len(preds) == len(experiments)

    candidate_count = 3
    candidates = strategy.ask(candidate_count)
    assert len(candidates) == candidate_count

    cost_data_model = [
        surr
        for surr in strategy.surrogate_specs.surrogates
        if surr.outputs.get_keys() == [strategy.fidelity_cost_output_key]
    ][0]
    task_input: ContinuousTaskInput = strategy.domain.inputs.get(ContinuousTaskInput)[0]
    assert isinstance(cost_data_model, LinearDeterministicSurrogate)
    # when the high fidelity is expensive, MFHVKG will (almost always) evaluate the low fidelity
    cost_data_model.coefficients[task_input.key] = 10.0
    candidate_lf = strategy.ask(1)
    assert candidate_lf[task_input.key].item() == task_input.lower_bound

    # when the high fidelity is cheap, MFHVKG will evaluate a higher fidelity
    task_input: ContinuousTaskInput = strategy.domain.inputs.get(ContinuousTaskInput)[0]
    cost_data_model.coefficients[task_input.key] = 0.01
    candidate_lf = strategy.ask(1)
    assert candidate_lf[task_input.key].item() > 0.1
