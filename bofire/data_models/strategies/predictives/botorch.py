from typing import Annotated, Optional, Type

from pydantic import Field, PositiveInt, root_validator, validator

from bofire.data_models.constraints.api import (
    Constraint,
    NonlinearEqualityConstraint,
    NonlinearInequalityConstraint,
)
from bofire.data_models.domain.api import Domain, Outputs
from bofire.data_models.enum import CategoricalEncodingEnum, CategoricalMethodEnum
from bofire.data_models.features.api import CategoricalDescriptorInput, CategoricalInput
from bofire.data_models.outlier_detection.api import OutlierDetections
from bofire.data_models.strategies.predictives.predictive import PredictiveStrategy
from bofire.data_models.surrogates.api import (
    BotorchSurrogates,
    MixedSingleTaskGPSurrogate,
    SingleTaskGPSurrogate,
)


def is_power_of_two(n):
    return (n != 0) and (n & (n - 1) == 0)


class BotorchStrategy(PredictiveStrategy):
    num_sobol_samples: PositiveInt = 512
    num_restarts: PositiveInt = 8
    num_raw_samples: PositiveInt = 1024
    descriptor_method: CategoricalMethodEnum = CategoricalMethodEnum.EXHAUSTIVE
    categorical_method: CategoricalMethodEnum = CategoricalMethodEnum.EXHAUSTIVE
    discrete_method: CategoricalMethodEnum = CategoricalMethodEnum.EXHAUSTIVE
    surrogate_specs: Optional[BotorchSurrogates] = None
    # outlier detection params
    outlier_detection_specs: Optional[OutlierDetections] = None
    min_experiments_before_outlier_check: PositiveInt = 1
    frequency_check: PositiveInt = 1
    # hyperopt params
    frequency_hyperopt: Annotated[int, Field(ge=0)] = 0  # 0 indicates no hyperopt
    folds: int = 5

    @classmethod
    def is_constraint_implemented(cls, my_type: Type[Constraint]) -> bool:
        """Method to check if a specific constraint type is implemented for the strategy

        Args:
            my_type (Type[Constraint]): Constraint class

        Returns:
            bool: True if the constraint type is valid for the strategy chosen, False otherwise
        """
        if my_type in [NonlinearInequalityConstraint, NonlinearEqualityConstraint]:
            return False
        return True

    @validator("num_sobol_samples")
    def validate_num_sobol_samples(cls, v):
        if is_power_of_two(v) is False:
            raise ValueError(
                "number sobol samples have to be of the power of 2 to increase performance"
            )
        return v

    @validator("num_raw_samples")
    def validate_num_raw_samples(cls, v):
        if is_power_of_two(v) is False:
            raise ValueError(
                "number raw samples have to be of the power of 2 to increase performance"
            )
        return v

    @root_validator(pre=False, skip_on_failure=True)
    def update_surrogate_specs_for_domain(cls, values):
        """Ensures that a prediction model is specified for each output feature"""
        values["surrogate_specs"] = BotorchStrategy._generate_surrogate_specs(
            values["domain"],
            values["surrogate_specs"],
        )
        # we also have to checke here that the categorical method is compatible with the chosen models
        if values["categorical_method"] == CategoricalMethodEnum.FREE:
            for m in values["surrogate_specs"].surrogates:
                if isinstance(m, MixedSingleTaskGPSurrogate):
                    raise ValueError(
                        "Categorical method FREE not compatible with a a MixedSingleTaskGPModel."
                    )
        #  we also check that if a categorical with descriptor method is used as one hot encoded the same method is
        # used for the descriptor as for the categoricals
        for m in values["surrogate_specs"].surrogates:
            keys = m.inputs.get_keys(CategoricalDescriptorInput)
            for k in keys:
                if m.input_preprocessing_specs[k] == CategoricalEncodingEnum.ONE_HOT:
                    if values["categorical_method"] != values["descriptor_method"]:
                        raise ValueError(
                            "One-hot encoded CategoricalDescriptorInput features has to be treated with the same method as categoricals."
                        )
        return values

    @root_validator(pre=False, skip_on_failure=True)
    def validate_outlier_detection_specs_for_domain(cls, values):
        """Ensures that a outlier_detection model is specified for each output feature"""
        if values["outlier_detection_specs"] is not None:
            values["outlier_detection_specs"]._check_compability(
                inputs=values["domain"].inputs, outputs=values["domain"].outputs
            )
        return values

    @staticmethod
    def _generate_surrogate_specs(
        domain: Domain,
        surrogate_specs: Optional[BotorchSurrogates] = None,
    ) -> BotorchSurrogates:
        """Method to generate model specifications when no model specs are passed
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
        existing_keys = (
            surrogate_specs.outputs.get_keys() if surrogate_specs is not None else []
        )
        non_exisiting_keys = list(set(domain.outputs.get_keys()) - set(existing_keys))
        _surrogate_specs = (
            surrogate_specs.surrogates if surrogate_specs is not None else []
        )
        for output_feature in non_exisiting_keys:
            if len(domain.inputs.get(CategoricalInput, exact=True)):
                _surrogate_specs.append(
                    MixedSingleTaskGPSurrogate(
                        inputs=domain.inputs,
                        outputs=Outputs(features=[domain.outputs.get_by_key(output_feature)]),  # type: ignore
                    )
                )
            else:
                _surrogate_specs.append(
                    SingleTaskGPSurrogate(
                        inputs=domain.inputs,
                        outputs=Outputs(features=[domain.outputs.get_by_key(output_feature)]),  # type: ignore
                    )
                )
        surrogate_specs = BotorchSurrogates(surrogates=_surrogate_specs)  # type: ignore
        surrogate_specs._check_compability(inputs=domain.inputs, outputs=domain.outputs)
        return surrogate_specs
