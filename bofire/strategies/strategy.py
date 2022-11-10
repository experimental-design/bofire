
from abc import abstractmethod
from typing import Optional, Type

import numpy as np
import pandas as pd
from pydantic import Field, validator
from pydantic.types import NonNegativeInt

from bofire.domain.constraints import Constraint
from bofire.domain.domain import Domain
from bofire.domain.features import Feature
from bofire.domain.util import BaseModel
from bofire.utils.reduce import AffineTransform, reduce_domain


class Strategy(BaseModel):
    """Base class for all strategies

    Args:
        BaseModel (pydantic.BaseModel): Pydantic base model

    Attributes:
        seed (NonNegativeInt, optional):                random seed to be used
        domain (Domain):                                the problem definition
        rng (np.random.Generator, optional):            the random generator used
        reduce (bool, optional):                        Boolean if irrelevant features or constraints should be ignored. Default is False.
        affine_transform (AffineTransform, optional):   Backward transformation to obtain original domain from reduced domain again
    """
    class Config:
        arbitrary_types_allowed = True

    seed: Optional[NonNegativeInt]
    domain: Domain
    rng: Optional[np.random.Generator]
    reduce: bool = False
    affine_transform: Optional[AffineTransform] = Field(default_factory=lambda: AffineTransform(equalities=[]))

    def __init__(self, domain: Domain, seed=None, reduce=False, *a, **kwa) -> None:
        """Constructor of strategy, reduces the domain if requested 
        """
        super().__init__(domain=domain, seed=seed, reduce=reduce, *a, **kwa)
        
        if self.reduce and self.is_reduceable(self.domain):
            self.domain, self.affine_transform = reduce_domain(self.domain)
        
        # we setup a random seed here
        if self.seed is None:
            self.seed = np.random.default_rng().integers(1000)
        self.rng = np.random.default_rng(self.seed)

        self._init_domain()   

    @validator("domain")
    def validate_feature_count(cls, domain: Domain):
        """Validator to ensure that at least one input and output feature is defined

        Args:
            domain (Domain): The domain to be used in the strategy

        Raises:
            ValueError: if no input feature is specified
            ValueError: if no output feature is specified

        Returns:
            Domain: the domain
        """
        if len(domain.input_features) == 0:
            raise ValueError("no input feature specified")
        if len(domain.output_features) == 0:
            raise ValueError("no output feature specified")
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
                raise ValueError(f"constraint `{type(constraint)}` is not implemented for strategy `{cls.__name__}`")
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
                raise ValueError(f"feature `{type(feature)}` is not implemented for strategy `{cls.__name__}`")
        return domain

    @abstractmethod
    def is_reduceable(self, domain: Domain) -> bool:
        """Function to check if the domain can be reduced

        Args:
            domain (Domain): The domain defining all input features

        Returns:
            Boolean: Boolean if the domain can be reduced
        """
        pass

    @abstractmethod
    def _init_domain(
        self,
    ) -> None:
        """Abstract method to allow for customized functions in the constructor of Strategy
        """
        pass

    def tell(
        self,
        experiments: pd.DataFrame,
        replace: bool = False,
    ) -> None:
        """This function passes new experimental data to the optimizer
        
        Irrelevant features are dropped if self.reduce is set to True 
        and the data is checked on validity before passed to the optimizer.

        Args:
            experiments (pd.DataFrame): DataFrame with experimental data
            replace (bool, optional): Boolean to decide if the experimental data should replace the former dataFrame or if the new experiments should be attached. Defaults to False.

        Raises:
            ValueError: if the domain is not specified
        """
        if self.domain is None:
            raise ValueError("domain is not initialized yet")
        if len(experiments) == 0:
            return
        experiments = self.affine_transform.drop_data(experiments)
        self.domain.validate_experiments(experiments)
        if replace or self.domain.experiments is None:
            self.domain.experiments = experiments
        else:
            self.domain.add_experiments(experiments)
        # TODO: check if provied constraints are implemented
        # TODO: validate that domain's output features match model_spec
        self._tell()

    @abstractmethod
    def _tell(
        self,
    ) -> None:
        """Abstract method to allow for customized tell functions in addition to self.tell()
        """
        pass

    def ask(
        self,
        candidate_count: int
    ) -> pd.DataFrame:
        """Function to generate new candidates

        Args:
            candidate_count (int): Number of candidates to be generated

        Raises:
            ValueError: if the number of generated candidates does not match the requested number

        Returns:
            pd.DataFrame: DataFrame with candidates (proposed experiments)
        """
        
        candidates = self._ask(candidate_count=candidate_count)

        self.domain.validate_candidates(candidates=candidates)
        candidates = self.affine_transform.augment_data(candidates)
        
        if len(candidates) != candidate_count:
            raise ValueError(f"expected {candidate_count} candidates, got {len(candidates)}")
        
        return candidates

    @abstractmethod
    def _ask(
        self,
        candidate_count: int,
    ) -> pd.DataFrame:
        """Abstract ask method to allow for customized ask functions in addition to self.ask()

        Args:
            candidate_count (int): Number of candidates to be generated

        Returns:
            pd.DataFrame: DataFrame with candidates (proposed experiments)
        """
        pass

    @abstractmethod
    def has_sufficient_experiments(
        self,
    ) -> bool:
        """Abstract method to check if sufficient experiments are provided

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
