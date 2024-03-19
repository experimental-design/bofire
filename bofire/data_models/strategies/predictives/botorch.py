import warnings
from abc import abstractmethod
from typing import Annotated, Literal, Optional, Type

from pydantic import Field, PositiveInt, field_validator, model_validator

from bofire.data_models.base import BaseModel
from bofire.data_models.constraints.api import (
    Constraint,
    LinearConstraint,
    NonlinearEqualityConstraint,
    NonlinearInequalityConstraint,
)
from bofire.data_models.domain.api import Domain, Outputs
from bofire.data_models.enum import CategoricalEncodingEnum, CategoricalMethodEnum
from bofire.data_models.features.api import CategoricalDescriptorInput, CategoricalInput
from bofire.data_models.outlier_detection.api import OutlierDetections
from bofire.data_models.strategies.predictives.predictive import PredictiveStrategy
from bofire.data_models.strategies.shortest_path import has_local_search_region
from bofire.data_models.surrogates.api import (
    BotorchSurrogates,
    MixedSingleTaskGPSurrogate,
    SingleTaskGPSurrogate,
)
from bofire.data_models.types import IntPowerOfTwo


class LocalSearchConfig(BaseModel):
    """LocalSearchConfigs provide a way to define how to switch between global
    acqf optimization in the global bounds and local acqf optimization in the local
    reference bounds.
    """

    type: str

    @abstractmethod
    def is_local_step(self, acqf_local: float, acqf_global: float) -> bool:
        """Abstract switching function between local and global acqf optimum.

        Args:
            acqf_local (float): Local acqf value.
            acqf_global (float): Global acqf value.

        Returns:
            bool: If true, do local step, else a step towards the global acqf maximum.
        """
        pass


class LSRBO(LocalSearchConfig):
    """LSRBO implements the local search region method published in.
    https://www.merl.com/publications/docs/TR2023-057.pdf

    Attributes:
        gamma (float): The switsching parameter between local and global optimization.
            Defaults to 0.1.
    """

    type: Literal["LSRBO"] = "LSRBO"
    gamma: Annotated[float, Field(ge=0)] = 0.1

    def is_local_step(self, acqf_local: float, acqf_global: float) -> bool:
        return acqf_local >= self.gamma


AnyLocalSearchConfig = LSRBO


class BotorchStrategy(PredictiveStrategy):
    # acqf optimizer params
    num_restarts: PositiveInt = 8
    num_raw_samples: IntPowerOfTwo = 1024
    maxiter: PositiveInt = 2000
    batch_limit: Optional[PositiveInt] = Field(default=None, validate_default=True)
    # encoding params
    descriptor_method: CategoricalMethodEnum = CategoricalMethodEnum.EXHAUSTIVE
    categorical_method: CategoricalMethodEnum = CategoricalMethodEnum.EXHAUSTIVE
    discrete_method: CategoricalMethodEnum = CategoricalMethodEnum.EXHAUSTIVE
    surrogate_specs: BotorchSurrogates = Field(
        default_factory=lambda: BotorchSurrogates(surrogates=[]), validate_default=True
    )
    # outlier detection params
    outlier_detection_specs: Optional[OutlierDetections] = None
    min_experiments_before_outlier_check: PositiveInt = 1
    frequency_check: PositiveInt = 1
    # hyperopt params
    frequency_hyperopt: Annotated[int, Field(ge=0)] = 0  # 0 indicates no hyperopt
    folds: int = 5
    # local search region params
    local_search_config: Optional[AnyLocalSearchConfig] = None

    @field_validator("batch_limit")
    @classmethod
    def validate_batch_limit(cls, batch_limit: int, info):
        batch_limit = min(
            batch_limit or info.data["num_restarts"], info.data["num_restarts"]
        )
        return batch_limit

    @model_validator(mode="after")
    def validate_local_search_config(self):
        if self.local_search_config is not None:
            if has_local_search_region(self.domain) is False:
                warnings.warn(
                    "`local_search_region` config is specified, but no local search region is defined in `domain`"
                )
            if (
                len(self.domain.constraints)
                - len(self.domain.constraints.get(LinearConstraint))
                > 0
            ):
                raise ValueError("LSR-BO only supported for linear constraints.")
        return self

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

    @model_validator(mode="after")
    def validate_surrogate_specs(self):
        """Ensures that a prediction model is specified for each output feature"""
        BotorchStrategy._generate_surrogate_specs(
            self.domain,
            self.surrogate_specs,
        )
        # we also have to checke here that the categorical method is compatible with the chosen models
        # categorical_method = (
        #   values["categorical_method"] if "categorical_method" in values else None
        # )
        if self.categorical_method == CategoricalMethodEnum.FREE:
            for m in self.surrogate_specs.surrogates:
                if isinstance(m, MixedSingleTaskGPSurrogate):
                    raise ValueError(
                        "Categorical method FREE not compatible with a a MixedSingleTaskGPModel."
                    )
        # we also check that if a categorical with descriptor method is used as one hot encoded the same method is
        # used for the descriptor as for the categoricals
        for m in self.surrogate_specs.surrogates:
            keys = m.inputs.get_keys(CategoricalDescriptorInput)
            for k in keys:
                input_proc_specs = (
                    m.input_preprocessing_specs[k]
                    if k in m.input_preprocessing_specs
                    else None
                )
                if input_proc_specs == CategoricalEncodingEnum.ONE_HOT:
                    if self.categorical_method != self.descriptor_method:
                        raise ValueError(
                            "One-hot encoded CategoricalDescriptorInput features has to be treated with the same method as categoricals."
                        )
        return self

    @model_validator(mode="after")
    def validate_outlier_detection_specs_for_domain(self):
        """Ensures that a outlier_detection model is specified for each output feature"""
        if self.outlier_detection_specs is not None:
            self.outlier_detection_specs._check_compability(
                inputs=self.domain.inputs, outputs=self.domain.outputs
            )
        return self

    @staticmethod
    def _generate_surrogate_specs(
        domain: Domain,
        surrogate_specs: BotorchSurrogates,
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
        existing_keys = surrogate_specs.outputs.get_keys()
        non_exisiting_keys = list(set(domain.outputs.get_keys()) - set(existing_keys))
        _surrogate_specs = surrogate_specs.surrogates
        for output_feature in non_exisiting_keys:
            if len(domain.inputs.get(CategoricalInput, exact=True)):
                _surrogate_specs.append(
                    MixedSingleTaskGPSurrogate(
                        inputs=domain.inputs,
                        outputs=Outputs(
                            features=[domain.outputs.get_by_key(output_feature)]  # type: ignore
                        ),
                    )
                )
            else:
                _surrogate_specs.append(
                    SingleTaskGPSurrogate(
                        inputs=domain.inputs,
                        outputs=Outputs(
                            features=[
                                domain.outputs.get_by_key(output_feature)  # type:ignore
                            ]
                        ),
                    )
                )
        surrogate_specs.surrogates = _surrogate_specs
        surrogate_specs._check_compability(inputs=domain.inputs, outputs=domain.outputs)
        return surrogate_specs
