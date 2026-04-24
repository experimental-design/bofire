from typing import Annotated, Union

from pydantic import Field

from bofire.data_models.transforms.drop_data import DropDataTransform
from bofire.data_models.transforms.manipulate_data import ManipulateDataTransform


AnyTransform = Annotated[
    Union[DropDataTransform, ManipulateDataTransform],
    Field(discriminator="type"),
]
