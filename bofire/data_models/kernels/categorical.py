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
