from typing import Literal

from pydantic import model_validator

from bofire.data_models.strategies.predictives.sobo import SoboStrategy
from bofire.data_models.surrogates.api import MultiTaskGPSurrogate


class MultiFidelityStrategy(SoboStrategy):
    type: Literal["MultiFidelityStrategy"] = "MultiFidelityStrategy"

    fidelity_thresholds: list[float] | float = 0.1

    @model_validator(mode="after")
    def validate_multitask_allowed(self):
        """Overwrites BotorchSurrogate.validate_multitask_allowed, as multiple tasks are allowed."""
        return self

    @model_validator(mode="after")
    def validate_only_multitask(self):
        """Ensures that all surrogates are multitask models"""
        if not all(
            isinstance(m, MultiTaskGPSurrogate) for m in self.surrogate_specs.surrogates
        ):
            raise ValueError(f"Must use a MultiTaskGPSurrogate with {self.type}.")
        return self
