from pydantic import ValidationError

import bofire.data_models.kernels.api as kernels
from bofire.data_models.priors.api import (
    GammaPrior,
    LogNormalPrior,
    NonTransformedInterval,
)
from tests.bofire.data_models.specs.prior_constraints import specs as prior_constraints
from tests.bofire.data_models.specs.priors import specs as priors
from tests.bofire.data_models.specs.specs import Specs


specs = Specs([])

specs.add_valid(
    kernels.PositiveIndexKernel,
    lambda: {
        "num_categories": 5,
        "rank": 1,
        "prior": None,
        "var_constraint": None,
        "task_prior": None,
        "diag_prior": None,
        "normalize_covar_matrix": False,
        "target_task_index": 0,
        "unit_scale_for_target": True,
        "features": None,
    },
)

specs.add_valid(
    kernels.PositiveIndexKernel,
    lambda: {
        "num_categories": 10,
        "rank": 3,
        "prior": priors.valid(GammaPrior).obj().model_dump(),
        "var_constraint": prior_constraints.valid(NonTransformedInterval)
        .obj()
        .model_dump(),
        "task_prior": priors.valid(LogNormalPrior).obj().model_dump(),
        "diag_prior": priors.valid(GammaPrior).obj().model_dump(),
        "normalize_covar_matrix": True,
        "target_task_index": 0,
        "unit_scale_for_target": False,
        "features": None,
    },
)

specs.add_invalid(
    kernels.PositiveIndexKernel,
    lambda: {
        "num_categories": 5,
        "rank": 6,
        "prior": None,
        "var_constraint": None,
        "task_prior": None,
        "diag_prior": None,
        "normalize_covar_matrix": False,
        "target_task_index": 0,
        "unit_scale_for_target": True,
        "features": None,
    },
    error=ValueError,
    message="rank must be less than or equal to num_categories",
)

specs.add_invalid(
    kernels.PositiveIndexKernel,
    lambda: {
        "num_categories": 1,
        "rank": 1,
        "prior": None,
        "var_constraint": None,
        "task_prior": None,
        "diag_prior": None,
        "normalize_covar_matrix": False,
        "target_task_index": 0,
        "unit_scale_for_target": True,
        "features": None,
    },
    error=ValidationError,
    message="Input should be greater than or equal to 2",
)

specs.add_invalid(
    kernels.PositiveIndexKernel,
    lambda: {
        "num_categories": 5,
        "rank": 3,
        "prior": None,
        "var_constraint": None,
        "task_prior": None,
        "diag_prior": None,
        "normalize_covar_matrix": False,
        "target_task_index": -1,
        "unit_scale_for_target": True,
        "features": None,
    },
    error=ValidationError,
    message="Input should be greater than or equal to 0",
)

specs.add_invalid(
    kernels.PositiveIndexKernel,
    lambda: {
        "num_categories": 5,
        "rank": 3,
        "prior": None,
        "var_constraint": None,
        "task_prior": None,
        "diag_prior": None,
        "normalize_covar_matrix": False,
        "target_task_index": 5,
        "unit_scale_for_target": True,
        "features": None,
    },
    error=ValidationError,
    message="target_task_index must be less than num_categories-1",
)

specs.add_valid(
    kernels.IndexKernel,
    lambda: {
        "num_categories": 5,
        "rank": 1,
        "prior": None,
        "var_constraint": None,
        "features": None,
    },
)

specs.add_valid(
    kernels.IndexKernel,
    lambda: {
        "num_categories": 10,
        "rank": 3,
        "prior": priors.valid(GammaPrior).obj().model_dump(),
        "var_constraint": prior_constraints.valid(NonTransformedInterval)
        .obj()
        .model_dump(),
        "features": None,
    },
)

specs.add_invalid(
    kernels.IndexKernel,
    lambda: {
        "num_categories": 5,
        "rank": 6,
        "prior": None,
        "var_constraint": None,
        "features": None,
    },
    error=ValueError,
    message="rank must be less than or equal to num_categories",
)

specs.add_invalid(
    kernels.IndexKernel,
    lambda: {
        "num_categories": 1,
        "rank": 1,
        "prior": None,
        "var_constraint": None,
        "features": None,
    },
    error=ValidationError,
    message="Input should be greater than or equal to 2",
)

specs.add_valid(
    kernels.HammingDistanceKernel,
    lambda: {
        "ard": True,
        "features": None,
        "lengthscale_prior": None,
        "lengthscale_constraint": None,
    },
)

specs.add_valid(
    kernels.HammingDistanceKernel,
    lambda: {
        "ard": True,
        "features": ["x_cat_1", "x_cat_2"],
        "lengthscale_prior": None,
        "lengthscale_constraint": None,
    },
)
specs.add_valid(
    kernels.WassersteinKernel,
    lambda: {
        "squared": False,
        "lengthscale_prior": priors.valid(GammaPrior).obj().model_dump(),
    },
)
specs.add_valid(
    kernels.LinearKernel,
    lambda: {
        "variance_prior": priors.valid(GammaPrior).obj().model_dump(),
        "features": None,
    },
)
specs.add_valid(
    kernels.MaternKernel,
    lambda: {
        "ard": True,
        "nu": 2.5,
        "features": None,
        "lengthscale_prior": priors.valid().obj().model_dump(),
        "lengthscale_constraint": prior_constraints.valid(NonTransformedInterval)
        .obj()
        .model_dump(),
    },
)
specs.add_invalid(
    kernels.MaternKernel,
    lambda: {
        "ard": True,
        "nu": 5,
        "lengthscale_prior": priors.valid().obj(),
        "lengthscale_constraint": prior_constraints.valid(NonTransformedInterval)
        .obj()
        .model_dump(),
        "features": None,
    },
    error=ValueError,
    message="nu expected to be 0.5, 1.5, or 2.5",
)
specs.add_valid(
    kernels.InfiniteWidthBNNKernel,
    lambda: {
        "depth": 3,
        "features": None,
    },
)

specs.add_valid(
    kernels.RBFKernel,
    lambda: {
        "ard": True,
        "lengthscale_prior": priors.valid().obj().model_dump(),
        "lengthscale_constraint": prior_constraints.valid(NonTransformedInterval)
        .obj()
        .model_dump(),
        "features": None,
    },
)
specs.add_valid(
    kernels.SphericalLinearKernel,
    lambda: {
        "ard": True,
        "lengthscale_prior": priors.valid().obj().model_dump(),
        "lengthscale_constraint": prior_constraints.valid(NonTransformedInterval)
        .obj()
        .model_dump(),
        "features": None,
        "bounds": (0.0, 1.0),
    },
)
specs.add_invalid(
    kernels.SphericalLinearKernel,
    lambda: {
        "ard": False,
        "lengthscale_prior": priors.valid().obj().model_dump(),
        "lengthscale_constraint": prior_constraints.valid(NonTransformedInterval)
        .obj()
        .model_dump(),
        "features": None,
        "bounds": (0.0, 1.0),
    },
    error=ValueError,
    message="Cannot determine number of dimensions. If ard=False then list of bounds should have length equal to the input dimension.",
)
specs.add_valid(
    kernels.ScaleKernel,
    lambda: {
        "base_kernel": specs.valid(kernels.LinearKernel).obj().model_dump(),
        "outputscale_prior": priors.valid(LogNormalPrior).obj().model_dump(),
        "outputscale_constraint": prior_constraints.valid(NonTransformedInterval)
        .obj()
        .model_dump(),
    },
)
specs.add_valid(
    kernels.AdditiveKernel,
    lambda: {
        "kernels": [
            specs.valid(kernels.LinearKernel).obj().model_dump(),
            specs.valid(kernels.MaternKernel).obj().model_dump(),
        ],
    },
)
specs.add_valid(
    kernels.MultiplicativeKernel,
    lambda: {
        "kernels": [
            specs.valid(kernels.LinearKernel).obj().model_dump(),
            specs.valid(kernels.MaternKernel).obj().model_dump(),
        ],
    },
)
specs.add_valid(
    kernels.TanimotoKernel,
    lambda: {
        "ard": True,
        "features": None,
    },
)
specs.add_valid(
    kernels.PolynomialFeatureInteractionKernel,
    lambda: {
        "max_degree": 2,
        "include_self_interactions": False,
        "kernels": [
            specs.valid(kernels.LinearKernel).obj().model_dump(),
            specs.valid(kernels.MaternKernel).obj().model_dump(),
        ],
        "outputscale_prior": priors.valid(LogNormalPrior).obj().model_dump(),
    },
)
specs.add_valid(
    kernels.WedgeKernel,
    lambda: {
        "base_kernel": specs.valid(kernels.LinearKernel).obj().model_dump(),
        "ard": True,
        "lengthscale_prior": priors.valid().obj().model_dump(),
        "lengthscale_constraint": prior_constraints.valid(NonTransformedInterval)
        .obj()
        .model_dump(),
        "angle_prior": priors.valid().obj().model_dump(),
        "radius_prior": priors.valid().obj().model_dump(),
        "conditions": [],
    },
)
