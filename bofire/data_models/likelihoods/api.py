from bofire.data_models.likelihoods.likelihood import (
    GaussianLikelihood,
    Likelihood,
    PairwiseLogitLikelihood,
    PairwiseProbitLikelihood,
)
from bofire.data_models.unions import tagged_union


AnyGaussianLikelihood = tagged_union(GaussianLikelihood)
AnyPairwiseLikelihood = tagged_union(PairwiseProbitLikelihood, PairwiseLogitLikelihood)
AnyLikelihood = AnyGaussianLikelihood | AnyPairwiseLikelihood
