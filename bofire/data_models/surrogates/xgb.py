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


class XGBoostSurrogate(Surrogate, TrainableSurrogate):
    type: Literal["XGBoostSurrogate"] = "XGBoostSurrogate"
    n_estimators: Annotated[int, Field(ge=1)] = 100
    max_depth: Annotated[int, Field(ge=0)] = 6
    max_leaves: Annotated[int, Field(ge=0)] = 0
    max_bin: Annotated[int, Field(ge=0)] = 256
    grow_policy: Literal["depthwise", "lossguide"] = "depthwise"
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

    @validator("input_preprocessing_specs", always=True)
    def validate_input_preprocessing_specs(cls, v, values):
        inputs = values["inputs"]
        categorical_keys = inputs.get_keys(CategoricalInput, exact=True)
        descriptor_keys = inputs.get_keys(CategoricalDescriptorInput, exact=True)
        for key in categorical_keys:
            if v.get(key, CategoricalEncodingEnum.ONE_HOT) not in [
                CategoricalEncodingEnum.ONE_HOT,
                CategoricalEncodingEnum.DUMMY,
                CategoricalEncodingEnum.ORDINAL,
            ]:
                raise ValueError(
                    "Botorch based models have to use one hot encodings for categoricals"
                )
            else:
                v[key] = CategoricalEncodingEnum.ONE_HOT
        for key in descriptor_keys:
            if v.get(key, CategoricalEncodingEnum.DESCRIPTOR) not in [
                CategoricalEncodingEnum.DESCRIPTOR,
                CategoricalEncodingEnum.ONE_HOT,
                CategoricalEncodingEnum.DUMMY,
                CategoricalEncodingEnum.ORDINAL,
            ]:
                raise ValueError(
                    "Botorch based models have to use one hot encodings or descriptor encodings for categoricals."
                )
            elif v.get(key) is None:
                v[key] = CategoricalEncodingEnum.DESCRIPTOR
        for key in inputs.get_keys(NumericalInput):
            if v.get(key) is not None:
                raise ValueError("Currently no numeric transforms are supported.")
        return v
