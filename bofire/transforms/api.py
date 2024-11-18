from typing import Dict, Type

import bofire.data_models.transforms.api as data_models
from bofire.transforms.drop_data import DropDataTransform
from bofire.transforms.manipulate_data import ManipulateDataTransform
from bofire.transforms.transform import Transform


TRANSFORM_MAP: Dict[Type[data_models.AnyTransform], Type[Transform]] = {
    data_models.DropDataTransform: DropDataTransform,
    data_models.ManipulateDataTransform: ManipulateDataTransform,
}


def map(data_model: data_models.AnyTransform) -> Transform:
    return TRANSFORM_MAP[data_model.__class__](data_model)
