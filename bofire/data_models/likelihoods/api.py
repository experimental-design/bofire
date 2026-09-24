from bofire.data_models.likelihoods.likelihood import GaussianLikelihood, Likelihood
from bofire.data_models.unions import tagged_union


AnyLikelihood = tagged_union(GaussianLikelihood)
