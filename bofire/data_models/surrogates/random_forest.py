from typing import Literal, Optional, Union

from pydantic import Field
from typing_extensions import Annotated

from bofire.data_models.surrogates.scaler import ScalerEnum
from bofire.data_models.surrogates.trainable_botorch import TrainableBotorchSurrogate


class RandomForestSurrogate(TrainableBotorchSurrogate):
    type: Literal["RandomForestSurrogate"] = "RandomForestSurrogate"

    # hyperparams passed down to `RandomForestRegressor`
    n_estimators: int = 100
    criterion: Literal[
        "squared_error",
        "absolute_error",
        "friedman_mse",
        "poisson",
    ] = "squared_error"
    max_depth: Optional[int] = None
    min_samples_split: Union[int, float] = 2
    min_samples_leaf: Union[int, float] = 1
    min_weight_fraction_leaf: float = 0.0
    max_features: Union[int, float, Literal["auto", "sqrt", "log2"]] = 1.0
    max_leaf_nodes: Optional[int] = None
    min_impurity_decrease: float = 0.0
    bootstrap: bool = True
    oob_score: bool = False
    random_state: Optional[int] = None
    ccp_alpha: Annotated[float, Field(ge=0)] = 0.0
    max_samples: Optional[Union[int, float]] = None
    scaler: ScalerEnum = ScalerEnum.IDENTITY
    output_scaler: ScalerEnum = ScalerEnum.IDENTITY
