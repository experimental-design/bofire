
from bofire.strategies.strategy import Strategy


class EntingStrategy(Strategy):
    """Strategy for randomly selecting new candidates.

    Provides a baseline strategy for benchmarks or for generating initial candidates.
    Uses PolytopeSampler or RejectionSampler, depending on the constraints.
    """

    def __init__(
        self,
        data_model,  #: data_models.EntingStrategy,
        **kwargs,
    ):
        super().__init__(data_model=data_model, **kwargs)
        self._init_sampler()
