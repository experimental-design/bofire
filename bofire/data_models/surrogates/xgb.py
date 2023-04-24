from typing import Annotated, Literal, Optional

from pydantic import Field

from bofire.data_models.surrogates.botorch import BotorchSurrogate


class XGBoostSurrogate(BotorchSurrogate):
    type: Literal["XGBoostSurrogate"] = "XGBoostSurrogate"
    n_estimators: int
    max_depth: Annotated[int, Field(ge=0)] = 6
    max_leaves: Annotated[int, Field(ge=0)] = 0
    max_bin: Annotated[int, Field(ge=0)] = 256
    grow_policy: Literal[0, 1] = 0
    learning_rate: Annotated[float, Field(gt=0.0, le=1.0)] = 0.3
    objective: Literal[
        "reg:squarederror",
        "reg:squaredlogerror",
        "reg:absoluteerror",
        "reg:pseudohubererror",
    ] = "reg:squarederror"
    booster: Literal["gbtree", "gblinear", "dart"] = "gbtree"
    n_jobs: Annotated[int, Field(gt=0)] = 1
    gamma: Annotated[float, Field(ge=0.0)] = 0.0
    min_child_weight: Annotated[float, Field(ge=0)] = 1.0
    max_delta_step: Annotated[float, Field(ge=0)] = 0.0
    subsample: Annotated[float, Field(gt=0, le=1)] = 1.0
    sampling_method: Literal["uniform", "gradient_based"] = "uniform"
    colsample_bytree: Annotated[float, Field(gt=0, le=1)] = 1.0
    colsample_bylevel: Annotated[float, Field(gt=0, le=1)] = 1.0
    colsample_bynode: Annotated[float, Field(gt=0, le=1)] = 1.0
    reg_alpha: Annotated[float, Field(ge=0)] = 0.0
    reg_lambda: Annotated[float, Field(ge=0)] = 1.0
    scale_pos_weight: Annotated[float, Field(ge=0)] = 1
    random_state: Optional[Annotated[int, Field(ge=0)]] = None
    num_parallel_tree: Annotated[int, Field(gt=0)] = 1
    # monotone_constraints
    # interaction_constraints
    # importance_type
    # validate_parameters
    # enable categorical
