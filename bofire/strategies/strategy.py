from abc import abstractmethod
from typing import Any, Dict, Optional, Tuple, Type

import numpy as np
import pandas as pd
from pydantic import BaseModel, validator
from pydantic.types import NonNegativeInt

from bofire.domain.constraints import Constraint
from bofire.domain.domain import Domain
from bofire.domain.features import Feature, OutputFeature
from bofire.domain.objectives import Objective
from bofire.utils.enum import CategoricalEncodingEnum


def validate_constraints(cls, domain: Domain):
    """Validator to ensure that all constraints defined in the domain are valid for the chosen strategy

    Args:
        domain (Domain): The domain to be used in the strategy

    Raises:
        ValueError: if a constraint is defined in the domain but is invalid for the strategy chosen

    Returns:
        Domain: the domain
    """
    for constraint in domain.constraints:
        if not cls.is_constraint_implemented(type(constraint)):
            raise ValueError(
                f"constraint `{type(constraint)}` is not implemented for strategy `{cls.__name__}`"  # type: ignore
            )
    return domain


def validate_features(cls, domain: Domain):
    """Validator to ensure that all features defined in the domain are valid for the chosen strategy

    Args:
        domain (Domain): The domain to be used in the strategy

    Raises:
        ValueError: if a feature type is defined in the domain but is invalid for the strategy chosen

    Returns:
        Domain: the domain
    """
    for feature in domain.inputs + domain.output_features:
        if not cls.is_feature_implemented(type(feature)):
            raise ValueError(
                f"feature `{type(feature)}` is not implemented for strategy `{cls.__name__}`"  # type: ignore
            )
    return domain


def validate_input_feature_count(cls, domain: Domain):
    """Validator to ensure that at least one input is defined.

    Args:
        domain (Domain): The domain to be used in the strategy

    Raises:
        ValueError: if no input feature is specified

    Returns:
        Domain: the domain
    """
    if len(domain.input_features) == 0:
        raise ValueError("no input feature specified")
    return domain


def validate_output_feature_count(cls, domain: Domain):
    """Validator to ensure that at least one output feature with attached objective is defined.

    Args:
        domain (Domain): The domain to be used in the strategy

    Raises:
        ValueError: if no output feature is specified
        ValueError: if not output feauture with an attached objective is specified

    Returns:
        Domain: the domain
    """
    if len(domain.output_features) == 0:
        raise ValueError("no output feature specified")
    if len(domain.outputs.get_by_objective(Objective)) == 0:
        raise ValueError("no output feature with objective specified")
    return domain


class Strategy(BaseModel):
    """Base class for all strategies

    Attributes:
        domain (Domain): The optimization domain ie. optimization problem defintion.
        seed (NonNegativeInt, optional): Random seed to be used, if no one defined, a seed is just sampled. Defaults to None.
    """

    class Config:
        arbitrary_types_allowed = True

    domain: Domain
    seed: Optional[NonNegativeInt] = None
    rng: Optional[np.random.Generator] = None

    _validate_constraints = validator("domain", allow_reuse=True)(validate_constraints)
    _validate_features = validator("domain", allow_reuse=True)(validate_features)
    _validate_input_feature_count = validator("domain", allow_reuse=True)(
        validate_input_feature_count
    )
    _validate_output_feature_count = validator("domain", allow_reuse=True)(
        validate_output_feature_count
    )

    def __init__(self, **data: Any):
        super().__init__(**data)
        # we setup a random seed here
        if self.seed is None:
            self.seed = np.random.default_rng().integers(1000)
        self.rng = np.random.default_rng(self.seed)

        self._init_domain()

    @validator("domain")
    def validate_objectives(cls, domain: Domain):
        """Validator to ensure that all objectives defined in the domain are valid for the chosen strategy

        Args:
            domain (Domain): The domain to be used in the strategy

        Raises:
            ValueError: if a objective type is defined in the domain but is invalid for the strategy chosen

        Returns:
            Domain: the domain
        """
        for feature in domain.outputs.get_by_objective(Objective):
            assert isinstance(feature, OutputFeature)
            assert feature.objective is not None
            if not cls.is_objective_implemented(type(feature.objective)):
                raise ValueError(
                    f"Objective `{type(feature.objective)}` is not implemented for strategy `{cls.__name__}`"  # type: ignore
                )
        return domain

    @abstractmethod
    def _init_domain(
        self,
    ) -> None:
        """Abstract method to allow for customized functions in the constructor of Strategy.

        Called at the end of `__init__`.
        """
        pass

    @property
    def experiments(self) -> Optional[pd.DataFrame]:
        """Property returning the experiments associated with the current strategy.

        Returns:
            pd.DataFrame: Experiments.
        """
        return self.domain.experiments

    @property
    def pending_candidates(self) -> Optional[pd.DataFrame]:
        """Candidates considered as pending.

        Returns:
            pd.DataFrame: pending candidates.
        """
        return self.domain.candidates

    def tell(
        self,
        experiments: pd.DataFrame,
        replace: bool = False,
    ):
        """This function passes new experimental data to the optimizer

        Args:
            experiments (pd.DataFrame): DataFrame with experimental data
            replace (bool, optional): Boolean to decide if the experimental data should replace the former dataFrame or if the new experiments should be attached. Defaults to False.
        """
        if len(experiments) == 0:
            return
        if replace:
            self.domain.set_experiments(experiments)
        else:
            self.domain.add_experiments(experiments)
        self._tell()

    @abstractmethod
    def _tell(
        self,
    ) -> None:
        """Abstract method to allow for customized tell functions in addition to self.tell()"""
        pass

    def ask(
        self,
        candidate_count: Optional[NonNegativeInt] = None,
        add_pending: bool = False,
        candidate_pool: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Function to generate new candidates

        Args:
            candidate_count (NonNegativeInt, optional): Number of candidates to be generated. If not provided, the number
                of candidates is determined automatically. Defaults to None.
            add_pending (bool, optional): If true the proposed candidates are added to the set of pending experiments. Defaults to False.
            candidate_pool (pd.DataFrame, optional): Pool of candidates from which a final set of candidates should be chosen. If not provided,
                pool independent candidates are provided. Defaults to None.


        Raises:
            ValueError: if candidate count is smaller than 1
            ValueError: if not enough experiments are available to execute the strategy
            ValueError: if the number of generated candidates does not match the requested number

        Returns:
            pd.DataFrame: DataFrame with candidates (proposed experiments)
        """
        if candidate_count is not None and candidate_count < 1:
            raise ValueError(
                f"Candidate_count has to be at least 1 but got {candidate_count}."
            )
        if not self.has_sufficient_experiments():
            raise ValueError(
                "Not enough experiments available to execute the strategy."
            )

        if candidate_pool is None:
            candidates = self._ask(candidate_count=candidate_count)
        else:
            self.domain.validate_candidates(candidate_pool, only_inputs=True)
            if candidate_count is not None:
                assert candidate_count <= len(
                    candidate_pool
                ), "Number of requested candidates is larger than the pool from which they should be chosen."
            candidates = self._choose_from_pool(candidate_pool, candidate_count)

        self.domain.validate_candidates(candidates=candidates)

        if candidate_count is not None:
            if len(candidates) != candidate_count:
                raise ValueError(
                    f"expected {candidate_count} candidates, got {len(candidates)}"
                )

        if add_pending:
            self.domain.add_candidates(candidates)

        return candidates

    @abstractmethod
    def _ask(
        self,
        candidate_count: Optional[NonNegativeInt] = None,
    ) -> pd.DataFrame:
        """Abstract ask method to allow for customized ask functions in addition to self.ask().

        Args:
            candidate_count (NonNegativeInt, optional): Number of candidates to be generated. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame with candidates (proposed experiments)
        """
        pass

    @abstractmethod
    def _choose_from_pool(
        self,
        candidate_pool: pd.DataFrame,
        candidate_count: Optional[NonNegativeInt] = None,
    ) -> pd.DataFrame:
        """Abstract method to implement how a strategy chooses a set of candidates from a candidate pool.

        Args:
            candidate_pool (pd.DataFrame): The pool of candidates from which the candidates should be chosen.
            candidate_count (Optional[NonNegativeInt], optional): Number of candidates to choose. Defaults to None.

        Returns:
            pd.DataFrame: The chosen set of candidates.
        """
        pass

    @abstractmethod
    def has_sufficient_experiments(
        self,
    ) -> bool:
        """Abstract method to check if sufficient experiments are available.

        Returns:
            bool: True if number of passed experiments is sufficient, False otherwise
        """
        pass

    @classmethod
    @abstractmethod
    def is_constraint_implemented(cls, my_type: Type[Constraint]) -> bool:
        """Abstract method to check if a specific constraint type is implemented for the strategy

        Args:
            my_type (Type[Constraint]): Constraint class

        Returns:
            bool: True if the constraint type is valid for the strategy chosen, False otherwise
        """
        pass

    @classmethod
    @abstractmethod
    def is_feature_implemented(cls, my_type: Type[Feature]) -> bool:
        """Abstract method to check if a specific feature type is implemented for the strategy

        Args:
            my_type (Type[Feature]): Feature class

        Returns:
            bool: True if the feature type is valid for the strategy chosen, False otherwise
        """
        pass

    @classmethod
    @abstractmethod
    def is_objective_implemented(cls, my_type: Type[Objective]) -> bool:
        """Abstract method to check if a objective type is implemented for the strategy

        Args:
            my_type (Type[Objective]): Objective class

        Returns:
            bool: True if the objective type is valid for the strategy chosen, False otherwise
        """
        pass


class PredictiveStrategy(Strategy):
    """Base class for all model based strategies.

    Provides abstract scaffold for fit, predict, and calc_acquistion methods.
    """

    is_fitted: bool = False

    def __init__(self, **data: Any):
        super().__init__(**data)
        if self.domain.num_experiments > 0:
            self.fit()
            self._tell()

    @property
    @abstractmethod
    def input_preprocessing_specs(self) -> Dict[str, CategoricalEncodingEnum]:
        pass

    def tell(
        self, experiments: pd.DataFrame, replace: bool = False, retrain: bool = True
    ):
        """This function passes new experimental data to the optimizer.

        Args:
            experiments (pd.DataFrame): DataFrame with experimental data
            replace (bool, optional): Boolean to decide if the experimental data should replace the former dataFrame or if the new experiments should be attached. Defaults to False.
            retrain (bool, optional): If True, model(s) are retrained when new experimental data is passed to the optimizer. Defaults to True.
        """
        # maybe unite the preprocessor here with the one of the parent tell
        if len(experiments) == 0:
            return
        if replace:
            self.domain.set_experiments(experiments)
        else:
            self.domain.add_experiments(experiments)
        if retrain:
            self.fit()
        # we have a seperate _tell here for things that are relevant when setting up the strategy but unrelated
        # to fitting the models like initializing the ACQF.
        self._tell()

    def predict(self, experiments: pd.DataFrame) -> pd.DataFrame:
        """Run predictions for the provided experiments. Only input features have to be provided.

        Args:
            experiments (pd.DataFrame): Experimental data for which predictions should be performed.

        Returns:
            pd.DataFrame: Dataframe with the predicted values.
        """
        if self.is_fitted is not True:
            raise ValueError("Model not yet fitted.")
        # TODO: validate also here the experiments but only for the input_columns
        # transformed = self.transformer.transform(experiments)
        transformed = self.domain.inputs.transform(
            experiments=experiments, specs=self.input_preproccesing_specs
        )
        preds, stds = self._predict(transformed)
        if stds is not None:
            predictions = pd.DataFrame(
                data=np.hstack((preds, stds)),
                columns=[
                    "%s_pred" % feat.key
                    for feat in self.domain.outputs.get_by_objective(Objective)
                ]
                + [
                    "%s_sd" % featkey
                    for featkey in self.domain.outputs.get_by_objective(Objective)
                ],
            )
        else:
            predictions = pd.DataFrame(
                data=preds,
                columns=[
                    "%s_pred" % feat.key
                    for feat in self.domain.outputs.get_by_objective(Objective)
                ],
            )
        return predictions

    @abstractmethod
    def _predict(self, experiments: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Abstract method in which the actual prediction is happening. Has to be overwritten."""
        pass

    def fit(self):
        """Fit the model(s) to the experimental data."""
        assert (
            self.experiments is not None and len(self.experiments) > 0
        ), "No fitting data available"
        self.domain.validate_experiments(self.experiments, strict=True)
        # transformed = self.transformer.fit_transform(self.experiments)
        self._fit(self.experiments)
        self.is_fitted = True

    @abstractmethod
    def _fit(self, experiments: pd.DataFrame):
        """Abstract method where the acutal prediction are occuring."""
        pass
