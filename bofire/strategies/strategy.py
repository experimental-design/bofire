import os
from abc import abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd
import torch
from botorch.utils.sampling import draw_sobol_samples, get_polytope_samples
from pydantic import Field, validator
from pydantic.class_validators import root_validator
from pydantic.types import (
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    conint,
    conlist,
)
from sklearn.model_selection import ParameterGrid

from bofire.domain.constraints import (
    Constraint,
    LinearConstraint,
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
)
from bofire.domain.domain import Domain
from bofire.domain.features import (
    CategoricalDescriptorInputFeature,
    CategoricalInputFeature,
    ContinuousInputFeature,
    ContinuousOutputFeature,
    ContinuousOutputFeature_woDesFunc,
    Feature,
    InputFeature,
    OutputFeature,
)
from bofire.domain.util import BaseModel
from bofire.strategies.botorch import tkwargs
from bofire.utils.reduce import AffineTransform, reduce_domain


class KernelEnum(Enum):
    """Enumeration class of all supported kernels

    Currently, RBF and matern kernel (1/2, 3/2 and 5/2) are implemented.
    """
    RBF = "RBF"
    MATERN_25 = "MATERN_25"
    MATERN_15 = "MATERN_15"
    MATERN_05 = "MATERN_05"


class ScalerEnum(Enum):
    """Enumeration class of supported scalers

    Currently, normalization and standardization are implemented.
    """
    NORMALIZE = "NORMALIZE"
    STANDARDIZE = "STANDARDIZE"

class CategoricalMethodEnum(Enum):
    """Enumeration class of supported methods how to handle categorical features

    Currently, exhaustive search and free relaxation are implemented.
    """
    EXHAUSTIVE = "EXHAUSTIVE"
    FREE = "FREE"
#    OEN = "OEN"

class CategoricalEncodingEnum(Enum):
    """Enumeration class of implemented categorical encodings

    Currently, one-hot and ordinal encoding are implemented.
    """
    ONE_HOT = "ONE_HOT"
    ORDINAL = "ORDINAL"

class DescriptorMethodEnum(Enum):
    """Enumeration class of implemented methods how to handle discrete descriptors

    Currently, exhaustive search and free relaxation are implemented.
    """
    EXHAUSTIVE = "EXHAUSTIVE"
    FREE = "FREE"
#    OEN = "OEN"

class DescriptorEncodingEnum(Enum):
    """Enumeration class how categorical features with descriptors should be encoded

    Categoricals with descriptors can be handled similar to categoricals, or the descriptors can be used.
    """
    DESCRIPTOR = "DESCRIPTOR"
    CATEGORICAL = "CATEGORICAL"

class ModelSpec(BaseModel):
    """Model specifications defining a model for the model-based strategies
    
    Attributes:
        output_feature (str):       output the model should predict
        input_features (List[str]): list of input feature keys to be used for the model
        kernel (KernelEnum):        the kernel to be used 
        ard (bool):                 boolean to switch automated relevance detection of input features on/off   
        scaler (ScalerEnum):        the scaling method to be used for the 
        name (str, optional):       the name is set in the strategy

    Raises:
        ValueError: when passed input features are not uniquely named

    """
    output_feature: str
    input_features: conlist(item_type=str, min_items=1)
    kernel: KernelEnum
    ard: bool
    scaler: ScalerEnum
    name: Optional[str] #is set in strategies

    @validator("input_features", allow_reuse=True)
    def validate_input_features(cls, v):
        """Validator to check if passed inout features are unique

        Args:
            v (List(str)): Input feature keys

        Raises:
            ValueError: when passed input features are not uniquely named

        Returns:
            List[str]: List with input feature keys
        """
        if len(v) != len(set(v)):
            raise ValueError("input features are not unique")
        return v

    def get(self, keyname: str, value: Optional[str]):
        """helper function to get the value of the named attribute

        Args:
            keyname (str): attribute name
            value (str, optional): value which is returned, when the named attribute does not exist

        Returns:
            _type_: value of the named attribute
        """
        return getattr(self, keyname, value)

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

    def __init__(self, domain: Domain, seed=42, reduce=False, *a, **kwa) -> None:
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

    def is_reduceable(self, domain):
        """Function to check if the domain can be reduced

        Args:
            domain (Domain): The domain defining all input features

        Returns:
            Boolean: Boolean if the domain can be reduced
        """
        if self.model_specs is None:
            return True
        else:
            return all([set(model_spec.input_features) == set(domain.get_feature_keys(InputFeature)) for model_spec in self.model_specs])   

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
        if replace or self.experiments is None:
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

class ModelPredictiveStrategy(Strategy):
    """Base class for model-based strategies

    Args:
        Strategy (Strategy): Strategy base class

    Attributes:
        model_specs (List[ModelSpec], optional): List of model specification classes to define the models to be used
    
    """
    model_specs: Optional[conlist(item_type=ModelSpec, min_items=1)]

    @root_validator(pre=False)
    def update_model_specs_for_domain(cls, values):
        """Ensures that a prediction model is specified for each output feature
        """
        if values["domain"] is not None:
            values["model_specs"] = ModelPredictiveStrategy._generate_model_specs(
                values["domain"],
                values["model_specs"],
            )
        return values
    
    @staticmethod
    def _generate_model_specs(
        domain: Domain,
        model_specs: List[ModelSpec] = None,
    ) -> List[ModelSpec]:
        """Method to generate model specifications when no model specs are passed

        As default specification, a 5/2 matern kernel with automated relevance detection and normalization of the input features is used.

        Args:
            domain (Domain): The domain defining the problem to be optimized with the strategy
            model_specs (List[ModelSpec], optional): List of model specification classes specifying the models to be used in the strategy. Defaults to None.

        Raises:
            KeyError: if there is a model spec for an unknown output feature
            KeyError: if a model spec has an unknown input feature
        Returns:
            List[ModelSpec]: List of model specification classes
        """
        input_features = domain.get_feature_keys(InputFeature)
        output_features = domain.get_feature_keys(OutputFeature, excludes=[ContinuousOutputFeature_woDesFunc])
        if model_specs is None:
            model_specs = []
        existing_specs = [
            model_spec.output_feature
            for model_spec in model_specs
        ]
        for key in existing_specs:
            if key not in output_features:
                raise KeyError(f"there is a model spec for an unknown output feature {key}")
        for model_spec in model_specs:
            for input_feature in model_spec.input_features:
                if input_feature not in input_features:
                    raise KeyError(f"model spec of {model_spec.output_feature} has an unknown input feature: {input_feature}")
        for output_feature in output_features:
            if output_feature in existing_specs:
                continue
            model_specs.append(ModelSpec(
                output_feature=output_feature,
                input_features=[*input_features],
                kernel=KernelEnum.MATERN_25,
                ard=True,
                scaler=ScalerEnum.NORMALIZE,
            ))
        assert len(model_specs) == len(output_features)
        return model_specs
    
    # todo overwrite tell

    def predict(self, experiments: pd.DataFrame):
        """Method to predict outputs for given experimental input values

        Args:
            experiments (pd.DataFrame): Experiments where a output should be predicted for

        Returns:
            pd.DataFrame: DataFrame with predictioncs for all output features
        """
        # TODO: validate also here the experiments but only for the input_columns
        transformed = self.transformer.transform(experiments)
        preds, stds = self._predict(transformed)
        if stds is not None:
            predictions = pd.DataFrame(
                data=np.hstack((preds, stds)),
                columns=[
                    "%s_pred" % featkey
                    for featkey in self.transformer.domain.get_feature_keys(
                        ContinuousOutputFeature, excludes=[ContinuousOutputFeature_woDesFunc]
                    )
                ]
                + [
                    "%s_sd" % featkey
                    for featkey in self.transformer.domain.get_feature_keys(
                        ContinuousOutputFeature, excludes=[ContinuousOutputFeature_woDesFunc]
                    )
                ],
            )
        else:
            predictions = pd.DataFrame(
                data=preds,
                columns=[
                    "%s_pred" % featkey
                    for featkey in self.transformer.domain.get_feature_keys(
                        ContinuousOutputFeature, excludes=[ContinuousOutputFeature_woDesFunc]
                    )
                ],
            )
        return predictions

    def calc_acquisition(self, experiments: pd.DataFrame, combined:bool =False):
        """Calculates the aquisition value for given experimental measurements

        Args:
            experiments (pd.DataFrame): Experimental data
            combined (bool, optional): Boolean if the returned torch.tensor should be unsqueezed. Defaults to False.

        Returns:
            _type_: _description_
        """
        transformed = self.transformer.transform(experiments)
        return self._calc_acquisition(transformed, combined=combined)

    @abstractmethod
    def _calc_acquisition(self, transformed:pd.DataFrame, combined:bool = False):
        """Calculates the aquisition value for the transformed experimental measurements

        Args:
            transformed (pd.DataFrame): Transformed experimental input data.
            combined (bool, optional): Boolean if the returned torch.tensor should be unsqueezed. Defaults to False.
        """
        pass

    @abstractmethod
    def _predict(self, experiments: pd.DataFrame):
        """Abstract method to allow for customized prediction function in extension to self.predict()

        Args:
            experiments (pd.DataFrame): Experimental data where the outputs should be predicted for
        """
        pass

    def fit(self):
        """Fits the model to be used in the strategy to experimental data
        """
        assert self.experiments is not None, "No fitting data available"
        self.domain.validate_experiments(self.experiments, strict=True)
        transformed = self.transformer.fit_transform(self.experiments)
        self._fit(transformed)

    @abstractmethod
    def _fit(self, transformed: pd.DataFrame):
        """Abstract method to allow for extension of self.fit()

        Args:
            transformed (pd.DataFrame): Transformed experimental data
        """
        pass

    @abstractmethod
    def get_candidate_log(self):
        """Abstract method for customized extensions of the returned candidate dataframe
        """
        # add here desired outputs for each candidate, e.g. acquisition values
        pass


class RandomStrategy(Strategy):
    """Strategy performing random sampling

    Attributes:
        use_sobol (Boolean, optional): Boolean to choose Sobol sampling instead of random sampling. Default is True.
        unit_scaled (Boolean, optional): Boolean if constraint coefficients should be scaled. Default is True.
    """

    use_sobol: bool = True
    unit_scaled: bool = True

    def _init_domain(
        self,
    ) -> None:
        """Extension of the strategy constructor setting a random seed for torch
        """
        torch.manual_seed(self.seed)
        return

    def _tell(
        self,
    ) -> None:
        """This function has no task in the random strategy
        """
        pass

    def get_bounds(self):
        """Method to get the lower and upper bounds of all continuous input features

        Returns:
            torch.tensor: Tensor with lower and upper bounds for all continuous input features
        """
        # only for countinuous ones and irgnore fixed features
        lower = [feat.lower_bound for feat in self.domain.get_features(ContinuousInputFeature) if feat.is_fixed()==False]
        upper = [feat.upper_bound for feat in self.domain.get_features(ContinuousInputFeature) if feat.is_fixed()==False]
        return torch.tensor([lower,upper]).to(**tkwargs)

    @staticmethod
    def get_linear_constraints(domain:Domain, constraint:LinearConstraint, unit_scaled:bool = False):
        """Returns a list of constraints which are of the passed contraint type

        Args:
            domain (Domain): The domain defining the optimization problem
            constraint (LinearConstraint): Constraint type which should be returned.
            unit_scaled (Boolean, optional): Boolean if constraint coefficients should be scaled. Default is False.

        Returns:
            List[List[torch.tensor]]: _description_
        """
        constraints = []
        for c in domain.get_constraints(constraint):
            indices = []
            coefficients = []
            lower = []
            upper = []
            rhs = 0.
            for i, featkey in enumerate(c.features):
                idx = domain.get_feature_keys(InputFeature).index(featkey)
                feat = domain.get_feature(featkey)
                if feat.is_fixed():
                    rhs -= feat.fixed_value()*c.coefficients[i]
                else:
                    lower.append(feat.lower_bound)
                    upper.append(feat.upper_bound)
                    indices.append(idx)
                    coefficients.append(c.coefficients[i])# if unit_scaled == False else c_scaled.coefficients[i])
            if unit_scaled:
                lower = np.array(lower)
                upper = np.array(upper)
                s = upper-lower
                scaled_coefficients = s*np.array(coefficients)
                constraints.append((
                    torch.tensor(indices),
                    torch.tensor(scaled_coefficients).to(**tkwargs),
                    rhs+c.rhs-np.sum(np.array(coefficients)*lower))
                )
            else:
                constraints.append((torch.tensor(indices),torch.tensor(coefficients).to(**tkwargs),rhs+c.rhs))
        return constraints

    def _ask(
        self,
        candidate_count: int,
    ) -> Tuple[pd.DataFrame, List[dict]]:
        """Generates random values depending on the domain input features.

        for each input feature:
        - row "{if}" with value
        - if continuous feature: float value
        - else: str value

        for each output feature:
        - row "{of}_pred" with predicted value (float)
        - row "{of}_sd" with standard deviation (float)
        - row "{of}_des" with desirability (float)

        Args:
            candidate_count (int): the number of desired candidates

        Returns:
            Tuple[pd.DataFrame, List[dict]]: A Tuple of a dataframe and a list of dictionaries with the generated candidates.
        """

        bounds = self.get_bounds()
        bounds_cpu = bounds.cpu()

        if len(self.domain.get_constraints(LinearConstraint)) == 0:
            if self.use_sobol:
                candidates = draw_sobol_samples(
                    bounds = bounds,
                    n=candidate_count,
                    q=1,).squeeze()
                    #seed=self.seed).squeeze()#.detach().numpy()
                    #seed=self.seed
                #)
            else:
                X_rnd_nlzd = torch.rand(candidate_count,bounds_cpu.shape[-1],dtype=bounds.dtype)
                candidates = bounds_cpu[0] + (bounds_cpu[1]-bounds_cpu[0]) * X_rnd_nlzd
        else:
            # now use the hit and run sampler     
            candidates = get_polytope_samples(
                n = candidate_count,
                bounds = torch.tensor([[0. for i in range(bounds.shape[1])],[1. for i in range(bounds.shape[1])]]).to(**tkwargs) if self.unit_scaled else bounds,
                inequality_constraints = self.get_linear_constraints(domain=self.domain, constraint=LinearInequalityConstraint, unit_scaled=self.unit_scaled),
                equality_constraints = self.get_linear_constraints(domain=self.domain, constraint=LinearEqualityConstraint, unit_scaled=self.unit_scaled),
                n_burnin = 1000,
                #thinning=200
            )
            # transform candidates back 
            if self.unit_scaled:
                candidates = bounds[0] + candidates*(bounds[1]-bounds[0])


        # check that the random generated candidates are not always the same
        if (candidates.unique(dim=0).shape[0] != candidate_count) and (candidate_count > 1):
            raise ValueError("Generated candidates are not unique!")

        # setup the output
        df_candidates = pd.DataFrame(
            data=np.nan,
            index=range(candidate_count),
            columns=self.domain.get_feature_keys(InputFeature)+ \
                [i+'_pred' for i in self.domain.get_feature_keys(OutputFeature, excludes=[ContinuousOutputFeature_woDesFunc])]+\
                [i+'_sd' for i in self.domain.get_feature_keys(OutputFeature, excludes=[ContinuousOutputFeature_woDesFunc])]+\
                [i+'_des' for i in self.domain.get_feature_keys(OutputFeature, excludes=[ContinuousOutputFeature_woDesFunc])]
            )

        free_continuals = [feat.key for feat in self.domain.get_features(ContinuousInputFeature) if feat.is_fixed()==False]    
        values = candidates.detach().numpy().reshape(candidate_count, len(free_continuals))
        df_candidates[free_continuals] = values 
        
        # setup the categoricals
        for i, feat in enumerate(self.domain.get_features(CategoricalInputFeature)):
            df_candidates[feat.key] = self.rng.choice(np.array(feat.get_allowed_categories()), candidate_count, replace=True)

        # setup the fixed values
        for i, feat in enumerate(self.domain.get_features(InputFeature)):
            if feat.is_fixed():
                df_candidates[feat.key] = feat.fixed_value()

        return df_candidates


    def has_sufficient_experiments(
        self,
    ) -> bool:
        """A random Strategy always has sufficient experiments.

        Returns:    
            bool: True
        """
        return True

    @classmethod
    def is_constraint_implemented(cls, my_type: Type[Constraint]) -> bool:
        """Method to check if a specific constraint type is implemented for the strategy

        Args:
            my_type (Type[Constraint]): Constraint class

        Returns:
            bool: False for NChooseKConstraint, True otherwise
        """
        if my_type == NChooseKConstraint:
            return False
        return True
    
    @classmethod
    def is_feature_implemented(cls, my_type: Type[Feature]) -> bool:
        """This function has no task in the random strategy
        """
        return True


class GridSearch(Strategy):
    """Strategy to conduct a grid search on the parameter space

    Attributes:
        resolutions (Dict): Dictionary containing the requested step width in the grid for each feature
        counter (Int, optional): Running variable for iterations tracking the last row of candidates in the candidate_grid dataFrame. Default is 0.
        candidate_grid (pd.DataFrame, optional): to be 
    """

    resolutions: dict
    counter: int = 0
    candidate_grid: Optional[pd.DataFrame]

    def _init_domain(
        self,
    ) -> None:
        """Extension of the strategy constructor creating a grid over the parameter space defined by domain

        Raises:
            ValueError: if no precision is provided for a feature
        """
        param_grid = {}
        for feat in self.domain.get_features(ContinuousInputFeature):
            if feat.key not in self.resolutions.keys():
                raise ValueError(f"no precision provided for feature {feat.key}")
            param_grid[feat.key]=np.linspace(feat.lower_bound, feat.upper_bound, self.resolutions[feat.key]).tolist()
        for feat in self.domain.get_features(CategoricalInputFeature):
            param_grid[feat.key]=feat.categories
        self.candidate_grid = pd.DataFrame(list(ParameterGrid(param_grid)))
        # apply linear inequality constraints
        for c in self.domain.get_constraints(LinearInequalityConstraint):
            self.candidate_grid = self.candidate_grid.query(c.__str__())
        self.candidate_grid = self.candidate_grid.reset_index(drop=True)
        for feat in self.domain.get_feature_keys(OutputFeature, excludes=[ContinuousOutputFeature_woDesFunc]):
            self.candidate_grid[f"{feat}_pred"] = np.nan
            self.candidate_grid[f"{feat}_sd"] = np.nan
            self.candidate_grid[f"{feat}_des"] = np.nan
        return

    @property
    def num_grid_candidates(self):
        """Returns the number of grid candidates

        Returns:
            Int: Number of grid candidates
        """
        return self.candidate_grid.shape[0]

    def _tell(
        self,
    ) -> None:
        """This function has no task in the gridsearch strategy
        """
        pass

    def has_sufficient_experiments(
        self,
    ) -> bool:
        """A gridsearch Strategy always has sufficient experiments.

        Returns:    
            bool: True
        """
        return True

    @classmethod
    def is_constraint_implemented(cls, my_type: Type[Constraint]) -> bool:
        """Method to check if a specific constraint type is implemented for the strategy

        Args:
            my_type (Type[Constraint]): Constraint class

        Returns:
            bool: False for NChooseKConstraint and LinearEqualityConstraint, True otherwise
        """
        if my_type == NChooseKConstraint:
            return False
        if my_type == LinearEqualityConstraint:
            return False
        return True
    
    @classmethod
    def is_feature_implemented(cls, my_type: Type[Feature]) -> bool:
        """This function has no task in the gridsearch strategy
        """
        return True

    def _ask(self, candidate_count: int):
        """Function to generate new candidates, customized extension to self.ask()

        Args:
            candidate_count (int): Number of candidates requested

        Raises:
            ValueError: If not enough candidates are left to deliver

        Returns:
            pd.DataFrame: DataFrame with candidate experiments
        """
        if self.counter + candidate_count > self.num_grid_candidates:
            raise ValueError("Not enough candidates left to deliver")
        candidates = self.candidate_grid.loc[self.counter:self.counter+candidate_count].copy().reset_index(drop=True)
        counter += candidate_count
       
        return candidates
