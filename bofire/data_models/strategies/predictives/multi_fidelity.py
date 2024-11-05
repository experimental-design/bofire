from typing import Literal

from pydantic import model_validator

from bofire.data_models.strategies.predictives.sobo import SoboStrategy


class MultiFidelityStrategy(SoboStrategy):
    type: Literal["MultiFidelityStrategy"] = "MultiFidelityStrategy"

    fidelity_thresholds: list[float] | float = 0.1
    # def __init__(self, **kwargs):
    #     super().__init__(**kwargs)

    #     # TODO: do this check properly, using pydantic
    #     assert all(isinstance(s, MultiTaskGPSurrogate) for s in self.surrogate_specs)

    @model_validator(mode="after")
    def validate_multitask_allowed(self):
        pass
