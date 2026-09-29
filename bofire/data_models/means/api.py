from bofire.data_models.means.mean import ConstantMean, Mean, TaskConstantMean
from bofire.data_models.unions import tagged_union


AnyMean = tagged_union(ConstantMean, TaskConstantMean)
