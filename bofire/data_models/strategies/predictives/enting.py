from typing import Any, Dict, Literal, Type

from entmoot.models.model_params import EntingParams
from pydantic import Field, PositiveFloat, PositiveInt

from bofire.data_models.constraints.api import (
    Constraint,
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
)
from bofire.data_models.features.api import (
    CategoricalDescriptorInput,
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
    DiscreteInput,
    Feature,
)
from bofire.data_models.objectives.api import (
    MaximizeObjective,
    MinimizeObjective,
    Objective,
)
from bofire.data_models.base import BaseModel
from bofire.data_models.strategies.predictives.predictive import PredictiveStrategy

class UncParams(BaseModel):
    beta: PositiveFloat = 1.96
    bound_coeff: float = 0.5
    acq_sense: Literal["exploration", "penalty"] = "exploration"
    dist_trafo: Literal["normal", "standard"] = "normal"
    dist_metric: Literal["euclidean_squared", "l1", "l2"] = "euclidean_squared"
    cat_metric: Literal["overlap", "of", "goodall4"] = "overlap"
        

class TrainParams(BaseModel):
    # lightgbm training hyperparameters
    objective: str = "regression"
    metric: str = "rmse"
    boosting: str = "gbdt"
    num_boost_round: PositiveInt = 100
    max_depth: PositiveInt = 3
    min_data_in_leaf: PositiveInt = 1
    min_data_per_group: PositiveInt = 1
    verbose: int = -1


class TreeTrainParams(BaseModel):
    train_params: "TrainParams" = Field(default_factory=lambda: TrainParams())
    train_lib: Literal["lgbm"] = "lgbm"


class EntingParams(BaseModel):
    """Contains parameters for a mean and uncertainty model."""
    unc_params: "UncParams" = Field(default_factory=lambda: UncParams())
    tree_train_params: "TreeTrainParams" = Field(default_factory=lambda: TreeTrainParams())


class EntingStrategy(PredictiveStrategy):
    type: Literal["EntingStrategy"] = "EntingStrategy"
    enting_params: EntingParams = Field(default_factory=EntingParams)
    solver_params: Dict[str, Any] = {}
    learn_from_candidates_coeff: float = 10.0

    @classmethod
    def is_constraint_implemented(cls, my_type: Type[Constraint]) -> bool:
        return my_type in [
            LinearEqualityConstraint,
            LinearInequalityConstraint,
            NChooseKConstraint,
        ]

    @classmethod
    def is_feature_implemented(cls, my_type: Type[Feature]) -> bool:
        return my_type in [
            CategoricalInput,
            DiscreteInput,
            CategoricalDescriptorInput,
            ContinuousInput,
            ContinuousOutput,
        ]

    @classmethod
    def is_objective_implemented(cls, my_type: Type[Objective]) -> bool:
        return my_type in [MinimizeObjective, MaximizeObjective]
