from typing import Annotated, Literal, Optional

from pydantic import Field, model_validator

from bofire.data_models.kernels.kernel import FeatureSpecificKernel
from bofire.data_models.priors.api import AnyPrior, AnyPriorConstraint
from bofire.data_models.priors.constraint import Positive


class CategoricalKernel(FeatureSpecificKernel):
    pass


class HammingDistanceKernel(CategoricalKernel):
    type: Literal["HammingDistanceKernel"] = "HammingDistanceKernel"
    ard: bool = True
    lengthscale_prior: Optional[AnyPrior] = None
    lengthscale_constraint: Optional[AnyPriorConstraint] = None


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

    Attributes:
        num_categories (int): Number of categorical values, must be ≥ 2.
        rank (int): Rank of the kernel approximation, must be ≥ 1 and ≤ num_categories.
                    Lower rank provides more regularization.
        prior (Optional[AnyPrior]): Prior distribution over kernel hyperparameters.
        var_constraint (Optional[AnyPriorConstraint]): Constraint on variance parameter,
                                                         defaults to Positive().
    Raises:
        ValueError: If rank > num_categories.
    """

    type: Literal["IndexKernel"] = "IndexKernel"
    num_categories: Annotated[int, Field(ge=2)]
    rank: Annotated[int, Field(ge=1)] = 1
    prior: Optional[AnyPrior] = None
    var_constraint: Optional[AnyPriorConstraint] = Positive()

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

    Attributes:
        num_categories (int): Number of categorical values, must be ≥ 2.
        rank (int): Rank of the kernel approximation, must be ≥ 1 and ≤ num_categories.
                    Lower rank provides more regularization.
        prior (Optional[AnyPrior]): Prior distribution over kernel hyperparameters.
        var_constraint (Optional[AnyPriorConstraint]): Constraint on variance parameter,
                                                         defaults to Positive().
        task_prior (Optional[AnyPrior]): Prior distribution over task-specific parameters.
        diag_prior (Optional[AnyPrior]): Prior distribution over diagonal elements.
        normalize_covar_matrix (bool): Whether to normalize the covariance matrix.
        target_task_index (int): Index of the target task for normalization to value 1,
                                default to 0 (first category).
        unit_scale_for_target (bool): Whether to use unit scale for the target task.
    Raises:
        ValueError: If rank > num_categories.
    """

    type: Literal["PositiveIndexKernel"] = "PositiveIndexKernel"
    num_categories: Annotated[int, Field(ge=2)]
    rank: Annotated[int, Field(ge=1)] = 1
    prior: Optional[AnyPrior] = None
    var_constraint: Optional[AnyPriorConstraint] = Positive()
    task_prior: Optional[AnyPrior] = None
    diag_prior: Optional[AnyPrior] = None
    normalize_covar_matrix: bool = False
    target_task_index: Annotated[int, Field(ge=0)] = 0
    unit_scale_for_target: bool = True

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
