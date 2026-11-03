from bofire.data_models.means.mean import ConstantMean, Mean, TaskwiseMean
from bofire.data_models.unions import tagged_union


AnyMean = tagged_union(ConstantMean)

# means of a multi-task GP: a plain mean is shared by all tasks
AnyMultiTaskMean = tagged_union(ConstantMean, TaskwiseMean)
