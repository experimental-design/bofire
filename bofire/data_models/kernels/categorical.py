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
    """Kernel measuring similarity as the number of categories two candidates share.

    Treats the categories as unordered and equally distant from one another, which is
    the right default unless there is structure among them that `IndexKernel` could
    learn instead.
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
    learns how similar the categories are from the data.
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
        description="Prior over the entries of the low-rank matrix. If not provided, "
        "the surrogate's default is used.",
    )
    var_constraint: Optional[AnyPriorConstraint] = Field(
        default=Positive(),
        description="Constraint on the per-category variance added on the diagonal, "
        "which keeps it non-negative.",
    )

    @model_validator(mode="after")
    def validate_rank_vs_categories(self):
        if self.rank is not None and self.rank > self.num_categories:
            raise ValueError("rank must be less than or equal to num_categories")
        return self


class PositiveIndexKernel(CategoricalKernel):
    r"""
    Many a times the IndexKernel is not positive definite. This kernel addresses this
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
        description="Prior over the entries of the triangular factor. If not provided, "
        "the surrogate's default is used.",
    )
    var_constraint: Optional[AnyPriorConstraint] = Field(
        default=Positive(),
        description="Constraint on the per-category variance added on the diagonal, "
        "which keeps it non-negative.",
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
