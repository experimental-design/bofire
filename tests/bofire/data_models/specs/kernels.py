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
    kernels.HammingDistanceKernel,
    lambda: {
        "ard": True,
        "features": None,
    },
)
specs.add_valid(
    kernels.HammingDistanceKernel,
    lambda: {
        "ard": True,
        "features": ["x_cat_1", "x_cat_2"],
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
