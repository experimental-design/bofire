import bofire.data_models.kernels.api as kernels
from tests.bofire.data_models.specs.priors import specs as priors
from tests.bofire.data_models.specs.specs import Specs

specs = Specs([])

specs.add_valid(
    kernels.HammingDistanceKernel,
    lambda: {
        "ard": True,
    },
)
specs.add_valid(
    kernels.LinearKernel,
    lambda: {"variance_prior": priors.valid().obj().model_dump()},
)
specs.add_valid(
    kernels.MaternKernel,
    lambda: {
        "ard": True,
        "nu": 2.5,
        "lengthscale_prior": priors.valid().obj().model_dump(),
    },
)
specs.add_invalid(
    kernels.MaternKernel,
    lambda: {
        "ard": True,
        "nu": 5,
        "lengthscale_prior": priors.valid().obj(),
    },
    error=ValueError,
    message="nu expected to be 0.5, 1.5, or 2.5",
)

specs.add_valid(
    kernels.RBFKernel,
    lambda: {
        "ard": True,
        "lengthscale_prior": priors.valid().obj().model_dump(),
    },
)
specs.add_valid(
    kernels.ScaleKernel,
    lambda: {
        "base_kernel": specs.valid(kernels.LinearKernel).obj().model_dump(),
        "outputscale_prior": priors.valid().obj().model_dump(),
    },
)
specs.add_valid(
    kernels.AdditiveKernel,
    lambda: {
        "kernels": [
            specs.valid(kernels.LinearKernel).obj().model_dump(),
            specs.valid(kernels.MaternKernel).obj().model_dump(),
        ]
    },
)
specs.add_valid(
    kernels.MultiplicativeKernel,
    lambda: {
        "kernels": [
            specs.valid(kernels.LinearKernel).obj().model_dump(),
            specs.valid(kernels.MaternKernel).obj().model_dump(),
        ]
    },
)
specs.add_valid(
    kernels.TanimotoKernel,
    lambda: {
        "ard": True,
    },
)
