from bofire.benchmarks.api import MOMFBraninCurrin
from bofire.data_models.enum import SamplingMethodEnum
from bofire.data_models.strategies.api import RandomStrategy as RandomStrategyDataModel
from bofire.strategies.api import MultiFidelityHVKGStrategy, RandomStrategy


def test_mfhvkg_fidelity_selection():
    benchmark = MOMFBraninCurrin()

    random_strategy = RandomStrategy(
        data_model=RandomStrategyDataModel(
            domain=benchmark.domain,
            fallback_sampling_method=SamplingMethodEnum.SOBOL,
            seed=42,
        ),
    )

    experiments = benchmark.f(random_strategy.ask(4), return_complete=True)

    strategy = MultiFidelityHVKGStrategy.make(
        domain=benchmark.domain,
    )

    strategy.tell(experiments)
    preds = strategy.predict(experiments)
    assert len(preds) == len(experiments)

    candidate_count = 3
    candidates = strategy.ask(candidate_count)
    assert len(candidates) == candidate_count
