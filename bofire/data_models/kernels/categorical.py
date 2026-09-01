from typing import Annotated, Literal, Optional

from pydantic import Field, model_validator

from bofire.data_models.kernels.kernel import (
    ARDKernel,
    FeatureSpecificKernel,
    LengthscaleKernel,
)
from bofire.data_models.priors.api import AnyPrior, AnyPriorConstraint
from bofire.data_models.priors.constraint import Positive


class CategoricalKernel(FeatureSpecificKernel):
    """Kernel acting on categorical inputs."""

    pass


class HammingDistanceKernel(ARDKernel, LengthscaleKernel, CategoricalKernel):
    r"""Kernel over categorical inputs based on the Hamming distance.

    $$
    k(\mathbf x, \mathbf x') = \exp\left(-\frac{d_H(\mathbf x, \mathbf x')}{\ell}\right)
    $$

    where $d_H$ counts the positions at which the two inputs hold different categories.
    Every pair of distinct categories is therefore equally far apart, which is the right
    assumption unless the categories have structure that `IndexKernel` could learn
    instead.
    """

    type: Literal["HammingDistanceKernel"] = "HammingDistanceKernel"


class IndexKernel(CategoricalKernel):
    r"""
    The Index kernel models categorical variables by assigning each
    category an index and learning a low-rank representation of the kernel matrix.
    This is particularly useful for ordered categorical variables or when categories have
    some inherent structure. Kernel is defined by a lookup table.
    Mathematically, the look up table is represented as:

    $$
    k(i, j) = \left(BB^\top + \text{diag}(\mathbf v) \right)_{i, j} # type: ignore
    $$

    where $B$ is a low-rank matrix, and $\mathbf v$ is a  non-negative vector.

    Unlike `HammingDistanceKernel`, which holds all categories equally distant, this
    learns from the data how similar the categories are.
    """

    type: Literal["IndexKernel"] = "IndexKernel"
    num_categories: Annotated[int, Field(ge=2)] = Field(
        description="Number of categories the kernel covers.",
    )
    rank: Annotated[int, Field(ge=1)] = Field(
        default=1,
        description="Rank of the learned similarity matrix, at most `num_categories`. "
        "A low rank forces the categories onto a few shared dimensions, which "
        "regularizes the fit; raising it allows a richer relationship at the cost of "
        "more parameters.",
    )
    prior: Optional[AnyPrior] = Field(
        default=None,
        description="Prior over the entries of $B$. If not provided, no prior is "
        "placed on them and they are fitted from the data alone.",
    )
    var_constraint: Optional[AnyPriorConstraint] = Field(
        default=Positive(),
        description="Bounds the diagonal entries $\\mathbf v$ are reparametrized into. "
        "Unlike `prior`, which only shifts the fitted value, this makes values outside "
        "the bounds unreachable; the default keeps the variances positive.",
    )

    @model_validator(mode="after")
    def validate_rank_vs_categories(self):
        if self.rank is not None and self.rank > self.num_categories:
            raise ValueError("rank must be less than or equal to num_categories")
        return self


class PositiveIndexKernel(CategoricalKernel):
    r"""
    The IndexKernel is often not positive definite. This kernel addresses that
    by using Cholesky decomposition with positive elements only. So, off diagonal
    elements are always positive and the diagonal elements are normalized to 1 for a
    target task. Mathematically, the kernel is defined as:

    $$
        k(i, j) = \frac{(LL^T)_{i,j}}{(LL^T)_{t,t}}
    $$

    where $L$ is a lower triangular matrix with positive elements and $t$ is the
    target_task_index.

    NOTE: This kernel should only be used when the correlation between different categories
    is expected to be positive.
    """

    type: Literal["PositiveIndexKernel"] = "PositiveIndexKernel"
    num_categories: Annotated[int, Field(ge=2)] = Field(
        description="Number of categories the kernel covers.",
    )
    rank: Annotated[int, Field(ge=1)] = Field(
        default=1,
        description="Rank of the learned similarity matrix, at most `num_categories`. "
        "A low rank forces the categories onto a few shared dimensions, which "
        "regularizes the fit; raising it allows a richer relationship at the cost of "
        "more parameters.",
    )
    prior: Optional[AnyPrior] = Field(
        default=None,
        description="Prior over the entries of $L$. If not provided, no prior is "
        "placed on them and they are fitted from the data alone.",
    )
    var_constraint: Optional[AnyPriorConstraint] = Field(
        default=Positive(),
        description="Bounds the diagonal entries $\\mathbf v$ are reparametrized into. "
        "Unlike `prior`, which only shifts the fitted value, this makes values outside "
        "the bounds unreachable; the default keeps the variances positive.",
    )
    task_prior: Optional[AnyPrior] = Field(
        default=None,
        description="Prior over the off-diagonal entries, which govern how strongly "
        "the categories are correlated with one another.",
    )
    diag_prior: Optional[AnyPrior] = Field(
        default=None,
        description="Prior over the diagonal entries, which govern how much variance "
        "each category has of its own.",
    )
    normalize_covar_matrix: bool = Field(
        default=False,
        description="Whether to rescale the whole matrix so that the correlations are "
        "read on a common scale.",
    )
    target_task_index: Annotated[int, Field(ge=0)] = Field(
        default=0,
        description="Index into the categories of the one treated as the target, whose "
        "diagonal entry is normalized to 1 so the others are expressed relative to it. "
        "Must be below `num_categories` - 1.",
    )
    unit_scale_for_target: bool = Field(
        default=True,
        description="Whether the target category is held at unit variance rather than "
        "having its own fitted.",
    )

    @model_validator(mode="after")
    def validate_rank_vs_categories(self):
        if self.rank is not None and self.rank > self.num_categories:
            raise ValueError("rank must be less than or equal to num_categories")
        return self

    @model_validator(mode="after")
    def validate_target_task_index(self):
        if (
            self.target_task_index is not None
            and self.target_task_index >= self.num_categories - 1
        ):
            raise ValueError("target_task_index must be less than num_categories-1")
        return self
