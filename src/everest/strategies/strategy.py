import os
from abc import abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd
import torch
from everest.domain.constraints import (ConcurrencyConstraint, Constraint,
                                        LinearConstraint,
                                        LinearEqualityConstraint,
                                        LinearInequalityConstraint)
from everest.domain.domain import Domain
from everest.domain.features import (CategoricalDescriptorInputFeature,
                                     CategoricalInputFeature,
                                     ContinuousInputFeature,
                                     ContinuousOutputFeature,
                                     ContinuousOutputFeature_woDesFunc,
                                     InputFeature, OutputFeature)
from everest.domain.util import BaseModel
from everest.strategies.botorch import tkwargs
from everest.utils.reduce import AffineTransform, reduce_domain
from pydantic import Field, validator
from pydantic.class_validators import root_validator
from pydantic.types import (NonNegativeFloat, NonNegativeInt, PositiveInt,
                            conint, conlist)
from sklearn.model_selection import ParameterGrid

from botorch.utils.sampling import draw_sobol_samples, get_polytope_samples


class KernelEnum(Enum):
    RBF = "RBF"
    MATERN_25 = "MATERN_25"
    MATERN_15 = "MATERN_15"
    MATERN_05 = "MATERN_05"


class ScalerEnum(Enum):
    NORMALIZE = "NORMALIZE"
    STANDARDIZE = "STANDARDIZE"

class CategoricalMethodEnum(Enum):
    EXHAUSTIVE = "EXHAUSTIVE"
    FREE = "FREE"
#    OEN = "OEN"

class CategoricalEncodingEnum(Enum):
    ONE_HOT = "ONE_HOT"
    ORDINAL = "ORDINAL"

class DescriptorMethodEnum(Enum):
    EXHAUSTIVE = "EXHAUSTIVE"
    FREE = "FREE"
#    OEN = "OEN"

class DescriptorEncodingEnum(Enum):
    DESCRIPTOR = "DESCRIPTOR"
    CATEGORICAL = "CATEGORICAL"

class ModelSpec(BaseModel):
    output_feature: str
    input_features: conlist(item_type=str, min_items=1)
    kernel: KernelEnum
    ard: bool
    scaler: ScalerEnum
    name: Optional[str] #is set in strategies

    @validator("input_features", allow_reuse=True)
    def validate_input_features(cls, v):
        if len(v) != len(set(v)):
            raise ValueError("input features are not unique")
        return v

    def get(self, keyname: str, value: Optional[str]):
        return getattr(self, keyname, value)

class Strategy(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    seed: Optional[NonNegativeInt]
    model_specs: Optional[conlist(item_type=ModelSpec, min_items=1)]
    domain: Optional[Domain]
    experiments: Optional[pd.DataFrame]
    rng: Optional[np.random.Generator]
    reduce: bool = False
    affine_transform: Optional[AffineTransform] = Field(default_factory=lambda: AffineTransform(equalities=[]))

    @staticmethod
    def _generate_model_specs(
        domain: Domain,
        model_specs: List[ModelSpec] = None,
    ) -> List[ModelSpec]:
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

    @validator("domain")
    def validate_feature_count(cls, domain: Domain):
        if len(domain.input_features) == 0:
            raise ValueError("no input feature specified")
        if len(domain.output_features) == 0:
            raise ValueError("no output feature specified")
        return domain

    @validator("domain")
    def validate_constraints(cls, domain: Domain):
        for constraint in domain.constraints:
            if not cls.is_implemented(type(constraint)):
                raise ValueError(f"constraint `{type(constraint)}` is not implemented for strategy `{cls.__name__}`")
        return domain

    @root_validator(pre=False)
    def update_model_specs_for_domain(cls, values):
        if values["domain"] is not None:
            values["model_specs"] = Strategy._generate_model_specs(
                values["domain"],
                values["model_specs"],
            )
        return values

    def is_reduceable(self, domain):
        if self.model_specs is None:
            return True
        else:
            return all([set(model_spec.input_features) == set(domain.get_feature_keys(InputFeature)) for model_spec in self.model_specs])

    def init_domain(
        self,
        domain: Domain,
    ) -> None:
        if self.reduce and self.is_reduceable(domain):
            self.domain, self.affine_transform = reduce_domain(domain)
        else:
            self.domain = domain
        # we setup a random seed here
        if self.seed is None:
            self.seed = np.random.default_rng().integers(1000)
        self.rng = np.random.default_rng(self.seed)

        self._init_domain()

    @abstractmethod
    def _init_domain(
        self,
    ) -> None:
        pass

    @classmethod
    def from_domain(cls, domain, **kwargs):
        strategy = cls(**kwargs)
        strategy.init_domain(domain)
        return strategy

    def tell(
        self,
        experiments: pd.DataFrame,
        replace: bool = False,
    ) -> None:
        if self.domain is None:
            raise ValueError("domain is not initialized yet")
        if len(experiments) == 0:
            return
        experiments = self.affine_transform.drop_data(experiments)
        self.domain.validate_experiments(experiments)
        if replace or self.experiments is None:
            self.experiments = experiments
        else:
            self.experiments = pd.concat((self.experiments, experiments),ignore_index=True) #self.experiments.append(experiments) # TODO: check that experiments is not NoneType
        # TODO: check if provied constraints are implemented
        # TODO: validate that domain's output features match model_spec
        self._tell()

    @abstractmethod
    def _tell(
        self,
    ) -> None:
        pass

    def ask(
        self,
        candidate_count: int,
        allow_insufficient_experiments: bool = False,
    ) -> Tuple[pd.DataFrame, List[dict]]:
        if not self.has_sufficient_experiments() and not allow_insufficient_experiments:
            raise ValueError("not enough experiments provided")
        if self.has_sufficient_experiments():
            candidates, configs = self._ask(candidate_count=candidate_count)
        elif allow_insufficient_experiments:
            random_strategy = RandomStrategy()
            random_strategy.init_domain(self.domain)
            if self.experiments is not None:
                random_strategy.tell(self.experiments)
            candidates, configs = random_strategy.ask(
                candidate_count=candidate_count,
                allow_insufficient_experiments=False,
            )
        else:
            raise ValueError("Not enough experiments available to execute the strategy. Set 'allow_insufficient_experiments=True'.")

        self.domain.validate_candidates(candidates=candidates)
        candidates = self.affine_transform.augment_data(candidates)
        if len(candidates) != len(configs):
            raise ValueError("candidates must have same length as configs")
        if len(candidates) != candidate_count:
            raise ValueError(f"expected {candidate_count} candidates, got {len(candidates)}")
        
        for config in configs:
            config["seed"] = self.seed
            config["requirements"] = dict(os.environ)
        return candidates, configs

    @abstractmethod
    def _ask(
        self,
        candidate_count: int,
    ) -> Tuple[pd.DataFrame, List[dict]]:
        pass

    @abstractmethod
    def has_sufficient_experiments(
        self,
    ) -> bool:
        pass

    @classmethod
    @abstractmethod
    def is_implemented(cls, my_type: Type[Constraint]) -> bool:
        pass

class ModelPredictiveStrategy(Strategy):
    # todo overwrite tell
    
    def predict(self, experiments: pd.DataFrame):
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
        transformed = self.transformer.transform(experiments)
        return self._calc_acquisition(transformed, combined=combined)

    @abstractmethod
    def _calc_acquisition(self, transformed:pd.DataFrame, combined:bool = False):
        pass

    @abstractmethod
    def _predict(self, experiments: pd.DataFrame):
        pass

    def fit(self):
        assert self.experiments is not None, "No fitting data available"
        self.domain.validate_experiments(self.experiments, strict=True)
        transformed = self.transformer.fit_transform(self.experiments)
        self._fit(transformed)

    @abstractmethod
    def _fit(self, transformed: pd.DataFrame):
        pass

    @abstractmethod
    def get_candidate_log(self):
        # add here desired outputs for each candidate, e.g. acquisition values
        pass


class RandomStrategy(Strategy):

    use_sobol: bool = True
    unit_scaled: bool = True

    def _init_domain(
        self,
    ) -> None:
        torch.manual_seed(self.seed)
        return

    def _tell(
        self,
    ) -> None:
        """This function has no task in the random strategy
        """
        pass

    def get_bounds(self):
        # only for countinuous ones and irgnore fixed features
        lower = [feat.lower_bound for feat in self.domain.get_features(ContinuousInputFeature) if feat.is_fixed()==False]
        upper = [feat.upper_bound for feat in self.domain.get_features(ContinuousInputFeature) if feat.is_fixed()==False]
        return torch.tensor([lower,upper]).to(**tkwargs)

    @staticmethod
    def get_linear_constraints(domain:Domain, constraint:LinearConstraint, unit_scaled:bool = False):
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

        configs = [
            {
                **self.dict(),
                "strategy": "RandomStrategy",
            }
            for _ in range(candidate_count)
        ]
        return df_candidates, configs


    def has_sufficient_experiments(
        self,
    ) -> bool:
        """A random Strategy always has sufficient experiments.

        Returns:    
            bool: True
        """
        return True

    @classmethod
    def is_implemented(cls, my_type: Type[Constraint]) -> bool:
        if my_type == ConcurrencyConstraint:
            return False
        return True
    
    def get_candidate_log(self):
        pass


class GridSearch(Strategy):

    resolutions: dict
    counter: int = 0
    candidate_grid: Optional[pd.DataFrame]

    def _init_domain(
        self,
    ) -> None:
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
    def is_implemented(cls, my_type: Type[Constraint]) -> bool:
        if my_type == ConcurrencyConstraint:
            return False
        if my_type == LinearEqualityConstraint:
            return False
        return True

    def get_candidate_log(self):
        pass

    def _ask(self, candidate_count: int):
        if self.counter + candidate_count > self.num_grid_candidates:
            raise ValueError("Not enough candidates left to deliver")
        candidates = self.candidate_grid.loc[self.counter:self.counter+candidate_count].copy().reset_index(drop=True)
        counter += candidate_count
        configs = [
            {
                "strategy": "GridSearch",
            }
            for _ in range(candidate_count)
        ]
        return candidates, configs
