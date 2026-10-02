from typing import Literal, Type

from pydantic import Field, model_validator

from bofire.data_models.acquisition_functions.api import (
    AnyPreferenceAcquisitionFunction,
    qEUBO,
)
from bofire.data_models.domain.api import Domain, Outputs
from bofire.data_models.features.api import ContinuousOutput, Feature, Output
from bofire.data_models.objectives.api import MaximizeObjective, Objective
from bofire.data_models.strategies.convergence_criteria.api import ConvergenceCriterion
from bofire.data_models.strategies.predictives.botorch import BotorchStrategy
from bofire.data_models.surrogates.api import PairwiseGPSurrogate


class PreferenceStrategy(BotorchStrategy):
    """Preferential Bayesian optimization with a pairwise GP."""

    type: Literal["PreferenceStrategy"] = "PreferenceStrategy"
    acquisition_function: AnyPreferenceAcquisitionFunction = Field(
        default_factory=qEUBO,
        description="Acquisition function used to propose alternatives. qEUBO "
        "defaults to a pair for a preference query; qLogNEI, qSR, and qUCB default "
        "to one candidate. All support larger batches. qSR and qEUBO evaluate the "
        "same expected maximum utility for the same batch.",
    )
    frequency_hyperopt: Literal[0] = Field(
        default=0,
        description="Cross-validation hyperparameter tuning is not supported for "
        "pairwise observations.",
    )

    @classmethod
    def _supports_pairwise_surrogates(cls) -> bool:
        return True

    @classmethod
    def _generate_single_surrogate_spec_for_output(
        cls, domain: Domain, output_feature: str
    ) -> PairwiseGPSurrogate:
        return PairwiseGPSurrogate(
            inputs=domain.inputs,
            outputs=Outputs(features=[domain.outputs.get_by_key(output_feature)]),
        )

    @model_validator(mode="after")
    def validate_pairwise_surrogate_specs(self):
        if len(self.domain.outputs) != 1 or not isinstance(
            self.domain.outputs[0], ContinuousOutput
        ):
            raise ValueError(
                "PreferenceStrategy requires exactly one continuous latent utility "
                "output."
            )
        if len(self.surrogate_specs.surrogates) != 1 or not isinstance(
            self.surrogate_specs.surrogates[0], PairwiseGPSurrogate
        ):
            raise ValueError(
                "PreferenceStrategy requires exactly one PairwiseGPSurrogate."
            )
        surrogate = self.surrogate_specs.surrogates[0]
        if surrogate.hyperconfig is not None:
            raise ValueError(
                "Hyperparameter tuning is not supported for preference surrogates."
            )
        return self

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
        # Criteria must be checked against latent utility and comparison feedback
        # before they can be enabled for preferential BO.
        return False
