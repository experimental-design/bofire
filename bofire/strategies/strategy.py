from abc import abstractmethod
from typing import Optional, Type, Union

import numpy as np
import pandas as pd
from pydantic import validate_arguments, validator
from pydantic.types import NonNegativeInt

from bofire.domain.constraints import Constraint
from bofire.domain.domain import Domain
from bofire.domain.features import Feature
from bofire.domain.objectives import Objective
from bofire.domain.util import BaseModel


class Strategy(BaseModel):
    """Base class for all strategies

    Attributes:
        domain (Domain): The optimization domain ie. optimization problem defintion.
        seed (NonNegativeInt, optional): Random seed to be used, if no one defined, a seed is just sampled. Defaults to None.
    """

    class Config:
        arbitrary_types_allowed = True

    domain: Domain
    seed: Optional[NonNegativeInt]
    rng: Optional[np.random.Generator]

    def __init__(self, domain: Domain, seed=None, *args, **kwargs) -> None:
        """Constructor of the strategy."""
        super().__init__(domain=domain, seed=seed, *args, **kwargs)

        # we setup a random seed here
        if self.seed is None:
            self.seed = np.random.default_rng().integers(1000)
        self.rng = np.random.default_rng(self.seed)

        self._init_domain()

    @validator("domain")
    def validate_feature_count(cls, domain: Domain):
        """Validator to ensure that at least one input and output feature with objective are defined

        Args:
            domain (Domain): The domain to be used in the strategy

        Raises:
            ValueError: if no input feature is specified
            ValueError: if no output feature is specified
            ValueError: if not output feauture with an attached objective is specified

        Returns:
            Domain: the domain
        """
        if len(domain.input_features) == 0:
            raise ValueError("no input feature specified")
        if len(domain.output_features) == 0:
            raise ValueError("no output feature specified")
        if len(domain.get_outputs_by_objective(Objective)) == 0:
            raise ValueError("no output feature with objective specified")
        return domain

    @validator("domain")
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
                    f"constraint `{type(constraint)}` is not implemented for strategy `{cls.__name__}`"
                )
        return domain

    @validator("domain")
    def validate_features(cls, domain: Domain):
        """Validator to ensure that all features defined in the domain are valid for the chosen strategy

        Args:
            domain (Domain): The domain to be used in the strategy

        Raises:
            ValueError: if a feature type is defined in the domain but is invalid for the strategy chosen

        Returns:
            Domain: the domain
        """
        for feature in domain.input_features + domain.output_features:
            if not cls.is_feature_implemented(type(feature)):
                raise ValueError(
                    f"feature `{type(feature)}` is not implemented for strategy `{cls.__name__}`"
                )
        return domain

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
        for feature in domain.get_outputs_by_objective(Objective):
            if not cls.is_objective_implemented(type(feature.objective)):
                raise ValueError(
                    f"Objective `{type(feature)}` is not implemented for strategy `{cls.__name__}`"
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
    def experiments(self) -> pd.DataFrame:
        """Property returning the experiments associated with the current strategy.

        Returns:
            pd.DataFrame: Experiments.
        """
        return self.domain.experiments

    @property
    def pending_candidates(self) -> pd.DataFrame:
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

    @validate_arguments
    def ask(
        self,
        candidate_count: Optional[NonNegativeInt] = None,
        add_pending: bool = False,
    ) -> pd.DataFrame:
        """Function to generate new candidates

        Args:
            candidate_count (NonNegativeInt, optional): Number of candidates to be generated. If not provided, the number
            of candidates is determined automatically. Defaults to None.

        Raises:
            ValueError: if not enough experiments are available to execute the strategy
            ValueError: if the number of generated candidates does not match the requested number

        Returns:
            pd.DataFrame: DataFrame with candidates (proposed experiments)
        """
        if not self.has_sufficient_experiments():
            raise ValueError(
                "Not enough experiments available to execute the strategy."
            )

        candidates = self._ask(candidate_count=candidate_count)

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
        """Abstract ask method to allow for customized ask functions in addition to self.ask()

        Args:
            candidate_count (NonNegativeInt, optional): Number of candidates to be generated. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame with candidates (proposed experiments)
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


class SamplingStrategy(Strategy):
    """Base class for all sampling based strategies which doe not use any previous experiments to compute the next set of candidates."""

    def _tell(self) -> None:
        return


class ModelPredictiveStrategy(Strategy):
    """Base class for all model based strategies.

    Provides abstract scaffold for fit, predict, and calc_acquistion methods.
    """

    def tell(
        self, experiments: pd.DataFrame, replace: bool = False, retrain: bool = True
    ):
        """This function passes new experimental data to the optimizer.

        Args:
            experiments (pd.DataFrame): DataFrame with experimental data
            replace (bool, optional): Boolean to decide if the experimental data should replace the former dataFrame or if the new experiments should be attached. Defaults to False.
            retrain (bool, optional): If True, model(s) are retrained when new experimental data is passed to the optimizer. Defaults to True.
        """
        super().tell(experiments, replace)
        if retrain:
            self.fit()

    def predict(self, experiments: pd.DataFrame) -> pd.DataFrame:
        """Run predictions for the provided experiments. Only input features have to be provided.

        Args:
            experiments (pd.DataFrame): Experimental data for which predictions should be performed.

        Returns:
            pd.DataFrame: Dataframe with the predicted values.
        """
        # TODO: validate also here the experiments but only for the input_columns
        transformed = self.transformer.transform(experiments)
        preds, stds = self._predict(transformed)
        if stds is not None:
            predictions = pd.DataFrame(
                data=np.hstack((preds, stds)),
                columns=[
                    "%s_pred" % featkey
                    for featkey in self.domain.get_output_keys_by_objective(Objective)
                ]
                + [
                    "%s_sd" % featkey
                    for featkey in self.domain.get_output_keys_by_objective(Objective)
                ],
            )
        else:
            predictions = pd.DataFrame(
                data=preds,
                columns=[
                    "%s_pred" % featkey
                    for featkey in self.domain.get_output_keys_by_objective(Objective)
                ],
            )
        return predictions

    def calc_acquisition(
        self, experiments: pd.DataFrame, combined: bool = False
    ) -> Union[float, pd.Series]:
        """Calculate the acquisition value(s) for a set of experiments/candidates

        Args:
            experiments (pd.DataFrame): Experiments/candidates for which the acquisiton value(s) should be calculated.
            combined (bool, optional): If combined the combined acqisition value is calculated, else the individual ones. Defaults to False.


        Returns:
            Union[float, pd.Series]: In case of `combined==True`, a float is returned else a pd.Series containing the individual acquisition values.
        """
        transformed = self.transformer.transform(experiments)
        return self._calc_acquisition(transformed, combined=combined)

    @abstractmethod
    def _calc_acquisition(self, transformed: pd.DataFrame, combined: bool = False):
        """Abstract method in which the actual acquistion function calculation is happening. Has to be overwritten."""
        pass

    @abstractmethod
    def _predict(self, experiments: pd.DataFrame):
        """Abstract method in which the actual prediction is happening. Has to be overwritten."""
        pass

    def fit(self):
        """Fit the model(s) to the experimental data."""
        assert (
            self.experiments is not None or len(self.experiments) == 0
        ), "No fitting data available"
        self.domain.validate_experiments(self.experiments, strict=True)
        transformed = self.transformer.fit_transform(self.experiments)
        self._fit(transformed)

    @abstractmethod
    def _fit(self, transformed: pd.DataFrame):
        """Abstract method where the acutal prediction are occuring."""
        pass
