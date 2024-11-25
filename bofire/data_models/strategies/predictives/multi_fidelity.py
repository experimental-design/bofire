from typing import List, Literal, Union

from pydantic import model_validator

from bofire.data_models.domain.api import Domain, Outputs
from bofire.data_models.strategies.predictives.sobo import SoboStrategy
from bofire.data_models.surrogates.api import BotorchSurrogates, MultiTaskGPSurrogate


class MultiFidelityStrategy(SoboStrategy):
    type: Literal["MultiFidelityStrategy"] = "MultiFidelityStrategy"

    fidelity_thresholds: Union[List[float], float] = 0.1

    @model_validator(mode="after")
    def validate_multitask_allowed(self):
        """Overwrites BotorchSurrogate.validate_multitask_allowed, as multiple tasks are allowed."""
        return self

    @model_validator(mode="after")
    def validate_surrogate_specs(self):
        """Ensures that a multi-task model is specified for each output feature"""
        MultiFidelityStrategy._generate_surrogate_specs(
            self.domain,
            self.surrogate_specs,
        )

        if not all(
            isinstance(m, MultiTaskGPSurrogate) for m in self.surrogate_specs.surrogates
        ):
            raise ValueError(f"Must use a MultiTaskGPSurrogate with {self.type}.")

        return self

    @staticmethod
    def _generate_surrogate_specs(
        domain: Domain,
        surrogate_specs: BotorchSurrogates,
    ) -> BotorchSurrogates:
        """Method to generate multi-task model specifications when no model specs are passed
        As default specification, a 5/2 matern kernel with automated relevance detection and normalization of the input features is used.
        Args:
            domain (Domain): The domain defining the problem to be optimized with the strategy
            surrogate_specs (List[ModelSpec], optional): List of model specification classes specifying the models to be used in the strategy. Defaults to None.
        Raises:
            KeyError: if there is a model spec for an unknown output feature
            KeyError: if a model spec has an unknown input feature
        Returns:
            List[ModelSpec]: List of model specification classes
        """
        existing_keys = surrogate_specs.outputs.get_keys()
        non_exisiting_keys = list(set(domain.outputs.get_keys()) - set(existing_keys))
        _surrogate_specs = surrogate_specs.surrogates
        for output_feature in non_exisiting_keys:
            _surrogate_specs.append(
                MultiTaskGPSurrogate(
                    inputs=domain.inputs,
                    outputs=Outputs(
                        features=[domain.outputs.get_by_key(output_feature)]
                    ),
                )
            )
        surrogate_specs.surrogates = _surrogate_specs
        surrogate_specs._check_compability(inputs=domain.inputs, outputs=domain.outputs)
        return surrogate_specs
