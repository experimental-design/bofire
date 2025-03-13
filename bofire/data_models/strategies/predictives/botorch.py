import warnings
from typing import Annotated, Optional, Type

from pydantic import Field, PositiveInt, model_validator

from bofire.data_models.constraints.api import (
    Constraint,
    InterpointConstraint,
    LinearConstraint,
)
from bofire.data_models.domain.api import Domain, Outputs
from bofire.data_models.enum import CategoricalEncodingEnum, CategoricalMethodEnum
from bofire.data_models.features.api import (
    CategoricalDescriptorInput,
    CategoricalInput,
    ContinuousInput,
    TaskInput,
)
from bofire.data_models.outlier_detection.api import OutlierDetections
from bofire.data_models.strategies.predictives.acqf_optimization import (
    AcquisitionOptimizer,
    BotorchOptimizer,
)
from bofire.data_models.strategies.predictives.predictive import PredictiveStrategy
from bofire.data_models.strategies.shortest_path import has_local_search_region
from bofire.data_models.surrogates.api import (
    BotorchSurrogates,
    MixedSingleTaskGPSurrogate,
    MultiTaskGPSurrogate,
    SingleTaskGPSurrogate,
)


class BotorchStrategy(PredictiveStrategy):
    # acquisition optimizer
    acquisition_optimizer: AcquisitionOptimizer = Field(
        default_factory=lambda: BotorchOptimizer()
    )

    surrogate_specs: BotorchSurrogates = Field(
        default_factory=lambda: BotorchSurrogates(surrogates=[]),
        validate_default=True,
    )
    # outlier detection params
    outlier_detection_specs: Optional[OutlierDetections] = None
    min_experiments_before_outlier_check: PositiveInt = 1
    frequency_check: PositiveInt = 1
    # hyperopt params
    frequency_hyperopt: Annotated[int, Field(ge=0)] = 0  # 0 indicates no hyperopt
    folds: int = 5

    @model_validator(mode="after")
    def validate_local_search_config(self):
        if not isinstance(self.acquisition_optimizer, BotorchOptimizer):
            return self

        if self.acquisition_optimizer.local_search_config is not None:
            if has_local_search_region(self.domain) is False:
                warnings.warn(
                    "`local_search_region` config is specified, but no local search region is defined in `domain`",
                )
            if (
                len(self.domain.constraints)
                - len(self.domain.constraints.get(LinearConstraint))
                > 0
            ):
                raise ValueError("LSR-BO only supported for linear constraints.")
        return self

    def is_constraint_implemented(self, my_type: Type[Constraint]) -> bool:
        """Method to check if a specific constraint type is implemented for the strategy. For optimizer-specific
        strategies, this is passed to the optimizer check.

        Args:
            my_type (Type[Constraint]): Constraint class

        Returns:
            bool: True if the constraint type is valid for the strategy chosen, False otherwise

        """
        return self.acquisition_optimizer.is_constraint_implemented(my_type)

    @model_validator(mode="after")
    def validate_interpoint_constraints(self):
        if self.domain.constraints.get(InterpointConstraint) and len(
            self.domain.inputs.get(ContinuousInput),
        ) != len(self.domain.inputs):
            raise ValueError(
                "Interpoint constraints can only be used for pure continuous search spaces.",
            )
        return self

    @model_validator(mode="after")
    def validate_surrogate_specs(self):
        """Ensures that a prediction model is specified for each output feature"""
        BotorchStrategy._generate_surrogate_specs(
            self.domain,
            self.surrogate_specs,
        )
        # we also have to check here that the categorical method is compatible with the chosen models
        # categorical_method = (
        #   values["categorical_method"] if "categorical_method" in values else None
        # )
        if isinstance(self.acquisition_optimizer, BotorchOptimizer):
            if (
                self.acquisition_optimizer.categorical_method
                == CategoricalMethodEnum.FREE
            ):
                for m in self.surrogate_specs.surrogates:
                    if isinstance(m, MixedSingleTaskGPSurrogate):
                        raise ValueError(
                            "Categorical method FREE not compatible with a a MixedSingleTaskGPModel.",
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
                                "One-hot encoded CategoricalDescriptorInput features has to be treated with the same method as categoricals.",
                            )
        return self

    @model_validator(mode="after")
    def validate_outlier_detection_specs_for_domain(self):
        """Ensures that a outlier_detection model is specified for each output feature"""
        if self.outlier_detection_specs is not None:
            self.outlier_detection_specs._check_compability(
                inputs=self.domain.inputs,
                outputs=self.domain.outputs,
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
                            features=[domain.outputs.get_by_key(output_feature)],
                        ),
                    ),
                )
            else:
                _surrogate_specs.append(
                    SingleTaskGPSurrogate(
                        inputs=domain.inputs,
                        outputs=Outputs(
                            features=[
                                domain.outputs.get_by_key(output_feature),  # type: ignore
                            ],
                        ),
                    ),
                )
        surrogate_specs.surrogates = _surrogate_specs
        surrogate_specs._check_compability(inputs=domain.inputs, outputs=domain.outputs)
        return surrogate_specs

    @model_validator(mode="after")
    def validate_multitask_allowed(self):
        """Ensures that if a multitask model is used there is only a single allowed task category"""
        if any(
            isinstance(m, MultiTaskGPSurrogate) for m in self.surrogate_specs.surrogates
        ):
            # find the task input
            task_input = self.domain.inputs.get(TaskInput, exact=True)
            # check if there is only one allowed task category
            assert (
                sum(task_input.features[0].allowed) == 1
            ), "Exactly one allowed task category must be specified for strategies with MultiTask models."
        return self
