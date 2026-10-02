from bofire.data_models.likelihoods.likelihood import (
    GaussianLikelihood,
    Likelihood,
    PairwiseLogitLikelihood,
    PairwiseProbitLikelihood,
)
from bofire.data_models.unions import tagged_union


AnyPairwiseLikelihood = tagged_union(PairwiseProbitLikelihood, PairwiseLogitLikelihood)
AnyLikelihood = tagged_union(
    GaussianLikelihood, PairwiseProbitLikelihood, PairwiseLogitLikelihood
)
