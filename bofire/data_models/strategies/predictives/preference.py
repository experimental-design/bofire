from typing import Literal, Type

from pydantic import Field, model_validator

from bofire.data_models.acquisition_functions.api import qEUBO
from bofire.data_models.constraints.api import Constraint
from bofire.data_models.features.api import ContinuousOutput, Feature, Output
from bofire.data_models.objectives.api import MaximizeObjective, Objective
from bofire.data_models.strategies.convergence_criteria.api import ConvergenceCriterion
from bofire.data_models.strategies.predictives.acqf_optimization import (
    AnyAcqfOptimizer,
    BotorchOptimizer,
)
from bofire.data_models.strategies.predictives.predictive import PredictiveStrategy
from bofire.data_models.surrogates.api import PairwiseGPSurrogate


class PreferenceStrategy(PredictiveStrategy):
    """Preferential Bayesian optimization with a pairwise GP and qEUBO."""

    type: Literal["PreferenceStrategy"] = "PreferenceStrategy"
    acquisition_function: qEUBO = Field(default_factory=qEUBO)
    acquisition_optimizer: AnyAcqfOptimizer = Field(default_factory=BotorchOptimizer)
    surrogate_spec: PairwiseGPSurrogate | None = None

    @model_validator(mode="after")
    def validate_domain_and_surrogate(self):
        if len(self.domain.outputs) != 1 or not isinstance(
            self.domain.outputs[0], ContinuousOutput
        ):
            raise ValueError(
                "PreferenceStrategy requires exactly one continuous latent utility "
                "output."
            )

        self.acquisition_optimizer.validate_domain(self.domain)

        if self.surrogate_spec is None:
            self.surrogate_spec = PairwiseGPSurrogate(
                inputs=self.domain.inputs,
                outputs=self.domain.outputs,
            )
        else:
            if self.surrogate_spec.inputs != self.domain.inputs:
                raise ValueError(
                    "The preference surrogate inputs must match the domain inputs."
                )
            if self.surrogate_spec.outputs != self.domain.outputs:
                raise ValueError(
                    "The preference surrogate output must match the domain output."
                )
        return self

    def is_constraint_implemented(self, my_type: Type[Constraint]) -> bool:
        return self.acquisition_optimizer.is_constraint_implemented(my_type)

    @classmethod
    def is_feature_implemented(cls, my_type: Type[Feature]) -> bool:
        if issubclass(my_type, Output):
            return my_type is ContinuousOutput
        return True

    @classmethod
    def is_objective_implemented(cls, my_type: Type[Objective]) -> bool:
        return my_type is MaximizeObjective

    @classmethod
    def is_criterion_implemented(cls, my_type: Type[ConvergenceCriterion]) -> bool:
        # Existing criteria consume observed output values, which are absent in
        # preferential BO. A preference-specific criterion can be added later.
        return False
