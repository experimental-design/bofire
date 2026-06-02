from bofire.data_models.transforms.drop_data import DropDataTransform
from bofire.data_models.transforms.manipulate_data import ManipulateDataTransform
from bofire.data_models.unions import tagged_union


AnyTransform = tagged_union(DropDataTransform, ManipulateDataTransform)
