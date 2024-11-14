import math
import warnings
from abc import abstractmethod
from typing import Annotated, Literal, Optional, Type, Union

import pandas as pd
from pydantic import Field, PositiveInt, field_validator, model_validator

from bofire.data_models.base import BaseModel
from bofire.data_models.constraints.api import (
    Constraint,
    InterpointConstraint,
    LinearConstraint,
    NonlinearEqualityConstraint,
    NonlinearInequalityConstraint,
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
from bofire.data_models.strategies.predictives.predictive import PredictiveStrategy
from bofire.data_models.strategies.shortest_path import has_local_search_region
from bofire.data_models.surrogates.api import (
    BotorchSurrogates,
    MixedSingleTaskGPSurrogate,
    MultiTaskGPSurrogate,
    SingleTaskGPSurrogate,
)
from bofire.data_models.types import IntPowerOfTwo


class TrustRegionConfig(BaseModel):
    """TrustRegionConfigs provide a way to define how to adapt local trust region
    constraints based on sucess.

    Args:
        length_init: Initial edge length
        length_min: Minimum edge length
        length_max: Maximum edge length
        lengthscale_adjustment_factor: Factor to adjust the lengthscale by when
            increasing or decreasing the dimensions of the trust region.
        fit_region_multiplier: Multiplier for the trust region dimensions to
            use when determining which points to fit the GP to.
        success_streak: Number of consecutive successes necessary to increase length
        failure_streak: Number of consecutive failures necessary to decrease length
        min_tr_size: The minimum number of points allowed in the trust region.
            If there are too few points in the TR, random samples will be used
            to refill the TR. NOTE this should be set to a low value. If using
            running experiments in parallel consider setting this to the
            candidate_count (batch_size).
        max_tr_size: The maximum number of points in a trust region. This can be
            used to avoid memory issues.
        experiment_batch_sizes: List of sizes of the experiment batches for
            each trust region. Used to determine whether or not a batch led to
            a success or failure and to prevent data sharing between trust regions.
        use_independent_tr: If true, each trust region will only fit to data
            from its own history. If false (default), all trust regions will
            fit to available data that overlaps the trust region.
    """

    type: str
    length_init: float = 0.8
    length_min: float = Field(default=1e-2, gt=0)
    length_max: float = 1.6
    lengthscale_adjustment_factor: float = Field(default=2.0, ge=1)
    fit_region_multiplier: float = Field(default=2.0, ge=1)
    min_tr_size: PositiveInt = 10
    max_tr_size: PositiveInt = 2048
    success_epsilon: float = Field(default=1e-3, gt=0)
    success_streak: PositiveInt = 3
    failure_streak: PositiveInt = 3
    success_counter: PositiveInt = 0
    failure_counter: PositiveInt = 0
    experiment_batch_sizes: list[list[PositiveInt]] = Field(
        default_factory=lambda: [[]]
    )
    use_independent_tr: bool = False

    length: float = Field(default=length_init)
    X_center_idx: int = -1

    @abstractmethod
    def update_trust_region(self, experiments: pd.DataFrame, domain: Domain) -> None:
        """Method to update the trust region based on the success of the optimization step.

        Args:
            experiments (pd.DataFrame): DataFrame containing experiments that were
                performed and their results.
            domain (Domain): domain containing the inputs and constraints.
        """

    def has_sufficient_experiments(
        self, experiments: pd.DataFrame, domain: Domain
    ) -> bool:
        """Do we have enough samples in the trust region to fit a model?

        Args:
            experiments (pd.DataFrame): DataFrame containing experiments that
                were performed and their results.
            domain (Domain): The domain defining the problem to be optimized
                with the strategy.
        """
        return (
            self.min_tr_size
            < self.get_trust_region_experiments(experiments, domain).shape[0]
        )

    def init_trust_region(self, experiments: pd.DataFrame, domain: Domain) -> None:
        """Method to initialize the trust region.

        Args:
            experiments (pd.DataFrame): DataFrame containing experiments that
                were performed and their results.
            domain (Domain): The domain defining the problem to be optimized
                with the strategy.
        """

        self.length = self.length_init

        Y_cols = domain.outputs.get_keys()
        if len(Y_cols) != 1:
            raise ValueError("TuRBO only supports single output optimization.")

        self.X_center_idx = experiments[Y_cols].idxmin().iloc[0]

    def get_trust_region_experiments(
        self, experiments: pd.DataFrame, domain: Domain, eps: float = 1e-8
    ) -> pd.DataFrame:
        """Method to get the experiments that are within the trust region.

        Args:
            experiments (pd.DataFrame): DataFrame containing experiments that
                were performed and their results.
            domain (Domain): The domain defining the problem to be optimized
                with the strategy.

        Returns:
            pd.DataFrame: DataFrame containing experiments that are within the
                trust region.
        """
        if self.use_independent_tr:
            tr_size = sum(self.experiment_batch_sizes[-1])
            experiments = experiments.iloc[-tr_size : len(experiments)]

        # TODO does X need to be normalized?
        X_cols = domain.inputs.get_keys()
        X = experiments[X_cols]

        X_center = X.loc[self.X_center_idx]

        # TODO scale box according to the kernel lengthscales if using ARD.
        # we seem to be limited by the abstraction here.

        tr_indicies = X[
            ((X - X_center).abs() - self.length / 2 <= eps).all(axis=1)
        ].index

        experiments = experiments.loc[tr_indicies]
        return experiments


class TuRBOConfig(TrustRegionConfig):
    """A trust region strategy for single objective bayesian optimization."""

    type: Literal["TuRBO"] = "TuRBO"

    def update_trust_region(self, experiments: pd.DataFrame, domain: Domain) -> None:
        """Method to update the trust region based on the success of the optimization step.

        Args:
            experiments (pd.DataFrame): DataFrame containing experiments that
                were performed and their results.
            domain (Domain): domain containing the inputs and constraints.
        """
        Y_cols = domain.outputs.get_keys()
        if len(Y_cols) != 1:
            raise ValueError("TuRBO only supports single output optimization.")

        self.X_center_idx = experiments[Y_cols].idxmin().iloc[0]
        Y_best = experiments.loc[self.X_center_idx][Y_cols].iloc[0]
        # Check that the experiments are success_epsilons better than the best
        # value, NOTE that the best value may be negative.
        if Y_best > self.best_value + self.success_epsilon * math.fabs(self.best_value):
            self.success_counter += 1
            self.failure_counter = 0
        else:
            self.success_counter = 0
            self.failure_counter += 1

        if self.success_counter == self.success_streak:  # Expand trust region
            self.length = min(
                self.lengthscale_adjustment_factor * self.length, self.length_max
            )
            self.success_counter = 0
        elif self.failure_counter == self.failure_streak:  # Shrink trust region
            self.length /= self.lengthscale_adjustment_factor
            self.failure_counter = 0

        self.best_value = min(self.best_value, Y_best)  # type: ignore
        if self.length < self.length_min:
            self.experiment_batch_sizes.append([])


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


class LSRBOConfig(LocalSearchConfig):
    """LSRBO implements the local search region method published in.
    https://www.merl.com/publications/docs/TR2023-057.pdf

    Attributes:
        gamma (float): The switching parameter between local and global optimization.
            Defaults to 0.1.
    """

    type: Literal["LSRBOConfig"] = "LSRBOConfig"
    gamma: Annotated[float, Field(ge=0)] = 0.1

    def is_local_step(self, acqf_local: float, acqf_global: float) -> bool:
        return acqf_local >= self.gamma


AnyLocalSearchConfig = Union[LSRBOConfig, LocalSearchConfig]
AnyTrustRegionConfig = Union[TuRBOConfig, TrustRegionConfig]


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
    # local search region params
    trust_region_config: Optional[AnyTrustRegionConfig] = None
    local_search_config: Optional[AnyLocalSearchConfig] = None

    @field_validator("batch_limit")
    @classmethod
    def validate_batch_limit(cls, batch_limit: int, info):
        batch_limit = min(
            batch_limit or info.data["num_restarts"],
            info.data["num_restarts"],
        )
        return batch_limit

    @model_validator(mode="after")
    def validate_local_search_config(self):
        if self.local_search_config is not None:
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
    def validate_exclusive_local_search_trust_region(self):
        # dont allow both local search and trust region to be used at the same time
        if (
            self.local_search_config is not None
            and self.trust_region_config is not None
        ):
            raise ValueError(
                "Local search and trust region optimization cannot be used at the same time."
            )
        return self

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
        if self.categorical_method == CategoricalMethodEnum.FREE:
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
