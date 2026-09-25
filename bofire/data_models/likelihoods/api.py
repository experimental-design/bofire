from bofire.data_models.likelihoods.likelihood import (
    GaussianLikelihood,
    Likelihood,
    TaskGaussianLikelihood,
)
from bofire.data_models.unions import tagged_union


AnyLikelihood = tagged_union(GaussianLikelihood, TaskGaussianLikelihood)
