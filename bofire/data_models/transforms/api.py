from typing import Union

from bofire.data_models.transforms.drop_data import DropDataTransform
from bofire.data_models.transforms.manipulate_data import ManipulateDataTransform


AnyTransform = Union[DropDataTransform, ManipulateDataTransform]
