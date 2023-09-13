from typing import Literal, Optional

from pydantic import Field, validator
from typing_extensions import Annotated

from bofire.data_models.enum import CategoricalEncodingEnum
from bofire.data_models.features.api import (
    CategoricalDescriptorInput,
    CategoricalInput,
    NumericalInput,
)
from bofire.data_models.surrogates.surrogate import Surrogate
from bofire.data_models.surrogates.trainable import TrainableSurrogate


class EntingSurrogate(Surrogate, TrainableSurrogate):
    type: Literal["EntingSurrogate"] = "EntingSurrogate"
    train_lib: Literal["lgbm"] = "lgbm"
    # mean model parameters
    objective: str = "regression"
    metric: str = "rmse"
    boosting: str = "gbdt"
    num_boost_round: Annotated[int, Field(ge=1)] = 100
    max_depth: Annotated[int, Field(ge=1)] = 3
    min_data_in_leaf: Annotated[int, Field(ge=1)] = 1
    min_data_per_group: Annotated[int, Field(ge=1)] = 1

    # uncertainty model parameters
    beta: Annotated[float, Field(gt=0)] = 1.96
    acq_sense: Literal["exploration", "penalty"] = "exploration"
    dist_trafo: Literal["normal", "standard"] = "normal"
    dist_metric: Literal["euclidean_squared", "l1", "l2"] = "euclidean_squared"
    cat_metric: Literal["overlap", "of", "goodall4"] = "overlap"



