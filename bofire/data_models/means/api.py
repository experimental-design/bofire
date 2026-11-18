from bofire.data_models.means.mean import ConstantMean, Mean, TaskwiseMean
from bofire.data_models.unions import tagged_union


AnyMean = tagged_union(ConstantMean, TaskwiseMean)

# the means a single-task GP takes
AnySingleTaskMean = tagged_union(ConstantMean)

# the means a multi-task GP takes: a plain mean is shared by all tasks
AnyMultiTaskMean = tagged_union(ConstantMean, TaskwiseMean)
