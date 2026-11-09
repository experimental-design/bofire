import gpytorch
import gpytorch.kernels
import pytest
import torch
from botorch.models.kernels import InfiniteWidthBNNKernel as BNNKernel
from botorch.models.kernels.categorical import CategoricalKernel
from botorch.utils.constraints import (
    LogTransformedInterval as BotorchLogTransformedInterval,
)
from gpytorch.kernels import IndexKernel as GpytorchIndexKernel

import bofire
import bofire.kernels.aggregation as aggregationKernels
import bofire.kernels.api as kernels
import bofire.kernels.conditional as conditionalKernels
import bofire.kernels.shape as shapeKernels
from bofire.data_models.constraints.condition import ThresholdCondition
from bofire.data_models.kernels.api import (
    AdditiveKernel,
    AdditiveMapSaasKernel,
    DownsamplingKernel,
    ExactWassersteinKernel,
    FeatureSpecificKernel,
    HammingDistanceKernel,
    IndexKernel,
    InfiniteWidthBNNKernel,
    LinearKernel,
    MaternKernel,
    MultiplicativeKernel,
    PolynomialFeatureInteractionKernel,
    PolynomialKernel,
    PositiveIndexKernel,
    RBFKernel,
    ScaleKernel,
    SphericalLinearKernel,
    TanimotoKernel,
    WassersteinKernel,
    WedgeKernel,
)
from bofire.data_models.priors.api import (
    CHEN_OUTPUTSCALE_PRIOR,
    THREESIX_SCALE_PRIOR,
    DimensionalityScaledGammaPrior,
    DimensionalityScaledLogNormalPrior,
    GammaPrior,
    LogTransformedInterval,
)
from bofire.kernels.mapper import _compute_active_dims
from tests.bofire.data_models.specs.api import Spec


EQUIVALENTS = {
    RBFKernel: gpytorch.kernels.RBFKernel,
    MaternKernel: gpytorch.kernels.MaternKernel,
    LinearKernel: gpytorch.kernels.LinearKernel,
    ScaleKernel: gpytorch.kernels.ScaleKernel,
    AdditiveKernel: gpytorch.kernels.AdditiveKernel,
    MultiplicativeKernel: gpytorch.kernels.ProductKernel,
    TanimotoKernel: bofire.kernels.fingerprint_kernels.tanimoto_kernel.TanimotoKernel,
    HammingDistanceKernel: CategoricalKernel,
    IndexKernel: GpytorchIndexKernel,
    PositiveIndexKernel: GpytorchIndexKernel,
    WassersteinKernel: shapeKernels.WassersteinKernel,
    ExactWassersteinKernel: shapeKernels.ExactWassersteinKernel,
    InfiniteWidthBNNKernel: BNNKernel,
    AdditiveMapSaasKernel: gpytorch.kernels.AdditiveKernel,
    PolynomialFeatureInteractionKernel: aggregationKernels.PolynomialFeatureInteractionKernel,
    WedgeKernel: conditionalKernels.WedgeKernel,
    SphericalLinearKernel: bofire.kernels.spherical_kernels.SphericalLinearKernel,
}


def test_map(kernel_spec: Spec):
    kernel = kernel_spec.cls(**kernel_spec.typed_spec())
    if (
        isinstance(kernel, (HammingDistanceKernel, DownsamplingKernel, TanimotoKernel))
        and kernel.features is not None
    ):
        return
    active_dims = list(range(5))
    mapper = None
    if isinstance(kernel, ExactWassersteinKernel):
        active_dims = list(range(6))

        def mapper(keys):
            return [int(key[1:]) for key in keys]

    gkernel = kernels.map(
        kernel,
        batch_shape=torch.Size(),
        active_dims=active_dims,
        features_to_idx_mapper=mapper,
    )
    assert isinstance(gkernel, EQUIVALENTS[kernel.__class__])


def test_map_scale_kernel():
    kernel = ScaleKernel(
        base_kernel=RBFKernel(),
        outputscale_prior=THREESIX_SCALE_PRIOR(),
        outputscale_constraint=LogTransformedInterval(
            lower_bound=0.01, upper_bound=10.0, initial_value=0.1
        ),
    )
    k = kernels.map(
        kernel,
        batch_shape=torch.Size(),
        active_dims=list(range(5)),
        features_to_idx_mapper=None,
    )
    assert hasattr(k, "outputscale_prior")
    assert isinstance(k.outputscale_prior, gpytorch.priors.GammaPrior)
    assert isinstance(k.raw_outputscale_constraint, BotorchLogTransformedInterval)
    kernel = ScaleKernel(base_kernel=RBFKernel())
    k = kernels.map(
        kernel,
        batch_shape=torch.Size(),
        active_dims=list(range(5)),
        features_to_idx_mapper=None,
    )
    assert hasattr(k, "outputscale_prior") is False


def test_map_scale_kernel_dimensionality_scaled_outputscale():
    # the outputscale prior is dimension-scaled, so the ScaleKernel mapper has to
    # forward the dimensionality d (= number of active dims) to the prior mapping.
    import math

    d = 7
    kernel = ScaleKernel(
        base_kernel=RBFKernel(),
        outputscale_prior=CHEN_OUTPUTSCALE_PRIOR(),
    )
    k = kernels.map(
        kernel,
        batch_shape=torch.Size(),
        active_dims=list(range(d)),
        features_to_idx_mapper=None,
    )
    assert isinstance(k.outputscale_prior, gpytorch.priors.GammaPrior)
    m = 0.4 * math.sqrt(d) + 4
    assert k.outputscale_prior.concentration == pytest.approx(m)
    assert k.outputscale_prior.rate == pytest.approx(1.0)


@pytest.mark.parametrize(
    "kernel",
    [
        LinearKernel(variance_prior=DimensionalityScaledGammaPrior()),
        PolynomialKernel(offset_prior=DimensionalityScaledGammaPrior()),
        PolynomialFeatureInteractionKernel(
            kernels=[RBFKernel(), MaternKernel()],
            max_degree=2,
            include_self_interactions=False,
            outputscale_prior=DimensionalityScaledGammaPrior(),
        ),
        IndexKernel(num_categories=3, rank=1, prior=DimensionalityScaledGammaPrior()),
        PositiveIndexKernel(
            num_categories=3, rank=1, task_prior=DimensionalityScaledGammaPrior()
        ),
    ],
)
def test_map_dimensionality_scaled_prior_on_kernel_prior_fields(kernel):
    # every kernel prior field accepts a dimensionality-scaled prior; the mapper threads
    # the dimensionality d to them, so mapping must not raise a missing-`d` error.
    k = kernels.map(
        kernel,
        batch_shape=torch.Size(),
        active_dims=list(range(5)),
        features_to_idx_mapper=None,
    )
    assert k is not None


def test_map_polynomial_kernel():
    kernel = PolynomialKernel(power=2, offset_prior=THREESIX_SCALE_PRIOR())
    k = kernels.map(
        kernel,
        batch_shape=torch.Size(),
        active_dims=list(range(5)),
        features_to_idx_mapper=None,
    )
    assert hasattr(k, "offset_prior")
    assert isinstance(k.offset_prior, gpytorch.priors.GammaPrior)
    kernel = PolynomialKernel(power=2)
    k = kernels.map(
        kernel,
        batch_shape=torch.Size(),
        active_dims=list(range(5)),
        features_to_idx_mapper=None,
    )
    assert hasattr(k, "offset_prior") is False


@pytest.mark.parametrize(
    "kernel, active_dims, expected_kernel",
    [
        (
            RBFKernel(
                ard=False,
                lengthscale_prior=GammaPrior(concentration=2.0, rate=0.15),
            ),
            list(range(5)),
            gpytorch.kernels.RBFKernel,
        ),
        (
            RBFKernel(ard=False),
            list(range(5)),
            gpytorch.kernels.RBFKernel,
        ),
        (RBFKernel(ard=True), list(range(5)), gpytorch.kernels.RBFKernel),
        (
            MaternKernel(
                ard=False,
                lengthscale_prior=GammaPrior(concentration=2.0, rate=0.15),
            ),
            list(range(5)),
            gpytorch.kernels.MaternKernel,
        ),
        (MaternKernel(ard=False), list(range(5)), gpytorch.kernels.MaternKernel),
        (MaternKernel(ard=True), list(range(5)), gpytorch.kernels.MaternKernel),
        (
            MaternKernel(ard=False, nu=2.5),
            list(range(5)),
            gpytorch.kernels.MaternKernel,
        ),
        (
            MaternKernel(ard=True, nu=1.5),
            list(range(5)),
            gpytorch.kernels.MaternKernel,
        ),
        (LinearKernel(), list(range(5)), gpytorch.kernels.LinearKernel),
    ],
)
def test_map_continuous_kernel(kernel, active_dims, expected_kernel):
    k = kernels.map(
        kernel,
        batch_shape=torch.Size(),
        active_dims=active_dims,
        features_to_idx_mapper=None,
    )
    assert isinstance(k, expected_kernel)
    if isinstance(kernel, LinearKernel):
        return
    if kernel.lengthscale_prior is not None:
        assert hasattr(k, "lengthscale_prior")
        assert isinstance(k.lengthscale_prior, gpytorch.priors.GammaPrior)
    else:
        assert hasattr(k, "lengthscale_prior") is False

    if kernel.ard is False:
        assert k.ard_num_dims is None
    else:
        assert k.ard_num_dims == len(active_dims)
    assert torch.eq(k.active_dims, torch.tensor(active_dims, dtype=torch.int64)).all()

    if isinstance(kernel, gpytorch.kernels.MaternKernel):
        assert kernel.nu == k.nu


@pytest.mark.parametrize(
    "kernel, active_dims, expected_kernel",
    [
        (
            TanimotoKernel(),
            list(range(5)),
            bofire.kernels.fingerprint_kernels.tanimoto_kernel.TanimotoKernel,
        ),
    ],
)
def test_map_molecular_kernel(kernel, active_dims, expected_kernel):
    k = kernels.map(
        kernel,
        batch_shape=torch.Size(),
        active_dims=active_dims,
        features_to_idx_mapper=None,
    )
    assert isinstance(k, expected_kernel)
    assert torch.eq(k.active_dims, torch.tensor(active_dims, dtype=torch.int64)).all()


def test_map_wasserstein_kernel():
    kernel = WassersteinKernel(
        squared=False,
        lengthscale_prior=GammaPrior(concentration=2.0, rate=0.15),
    )
    k = kernels.map(
        kernel,
        batch_shape=torch.Size(),
        active_dims=list(range(5)),
        features_to_idx_mapper=None,
    )
    assert isinstance(k, shapeKernels.WassersteinKernel)
    assert hasattr(k, "lengthscale_prior")
    assert isinstance(k.lengthscale_prior, gpytorch.priors.GammaPrior)
    assert k.squared is False
    kernel = WassersteinKernel(squared=True)
    k = kernels.map(
        kernel,
        batch_shape=torch.Size(),
        active_dims=list(range(5)),
        features_to_idx_mapper=None,
    )
    assert k.squared is True
    assert hasattr(k, "lengthscale_prior") is False


@pytest.mark.parametrize("kernel_cls", [WassersteinKernel, ExactWassersteinKernel])
@pytest.mark.parametrize("nodes", [2, 32])
@pytest.mark.parametrize(
    "prior, parameters",
    [
        (
            DimensionalityScaledLogNormalPrior(loc=0.3, scale=0.7),
            {"loc": 0.3, "scale": 0.7},
        ),
        (
            DimensionalityScaledGammaPrior(
                concentration=2.0, concentration_scaling=0.5, rate=0.8, rate_power=0.5
            ),
            {"concentration": 2.5, "rate": 0.8},
        ),
    ],
)
def test_shape_kernel_prior_independent_of_discretization(
    kernel_cls, nodes, prior, parameters
):
    kwargs = {}
    n_dims = nodes
    if kernel_cls is ExactWassersteinKernel:
        n_dims = 2 * nodes
        kwargs = {
            "x_keys": [f"x{i}" for i in range(nodes)],
            "y_keys": [f"x{i}" for i in range(nodes, n_dims)],
        }
    kernel = kernels.map(
        kernel_cls(lengthscale_prior=prior, **kwargs),
        batch_shape=torch.Size(),
        active_dims=list(range(n_dims)),
        features_to_idx_mapper=lambda keys: [int(key[1:]) for key in keys],
    )
    assert kernel.raw_lengthscale.shape == (1, 1)
    for name, value in parameters.items():
        assert getattr(kernel.lengthscale_prior, name).item() == pytest.approx(value)


def test_map_exact_wasserstein_kernel_auto_chunking():
    kernel = ExactWassersteinKernel(
        squared=False,
        x_keys=["x4", "x5"],
        y_keys=["x0", "x1", "x2", "x3"],
        prepend_x=[0.0],
        prepend_y=[],
        append_x=[1.0],
        append_y=[],
        normalize_y=100.0,
        normalize_x=True,
        order=2,
    )
    k = kernels.map(
        kernel,
        batch_shape=torch.Size(),
        active_dims=list(range(6)),
        features_to_idx_mapper=lambda keys: [int(key[1:]) for key in keys],
    )
    assert isinstance(k, shapeKernels.ExactWassersteinKernel)
    assert k.pair_chunk_size is None
    assert k.order == 2


@pytest.mark.parametrize("features", [None, ["y_end", "x_end", "y_start", "x_start"]])
def test_exact_wasserstein_coordinate_keys_preserve_order(features):
    columns = {"x_start": 4, "x_end": 1, "y_start": 5, "y_end": 2}

    def mapper(keys):
        # Feature mappers can sort their result by input order.
        return sorted(columns[key] for key in keys)

    spec = ExactWassersteinKernel(
        x_keys=["x_start", "x_end"],
        y_keys=["y_start", "y_end"],
        features=features,
    )
    kernel = kernels.map(
        spec,
        batch_shape=torch.Size(),
        active_dims=[5, 3, 0, 1, 4, 2],
        features_to_idx_mapper=mapper,
    ).double()
    kernel.lengthscale = 1.0
    # Columns 0 and 3 have no bearing on the curve. Both curves have x=[0, 1]
    # and their y-coordinates differ by 0.3 everywhere.
    x = torch.tensor(
        [[99.0, 1.0, 0.7, -10.0, 0.0, 0.2], [22.0, 1.0, 1.0, 0.0, 0.0, 0.5]],
        dtype=torch.double,
    )
    actual = kernel(x).to_dense()
    expected = torch.exp(-torch.tensor([[0.0, 0.3], [0.3, 0.0]], dtype=torch.double))
    torch.testing.assert_close(actual, expected)


def test_exact_wasserstein_coordinate_must_be_scalar():
    # One unique feature key can expand into several input columns.
    columns = {"ramp": [0, 1], "y": [2]}
    with pytest.raises(ValueError, match="Curve coordinate 'ramp'.*exactly one"):
        kernels.map(
            ExactWassersteinKernel(
                x_keys=["ramp"], y_keys=["y"], prepend_x=[0.0], prepend_y=[0.0]
            ),
            batch_shape=torch.Size(),
            active_dims=[0, 1, 2],
            features_to_idx_mapper=lambda keys: [
                i for key in keys for i in columns[key]
            ],
        )


def test_exact_wasserstein_coordinate_keys_require_mapper():
    with pytest.raises(RuntimeError, match="must be defined to resolve curve"):
        kernels.map(
            ExactWassersteinKernel(
                x_keys=["x"], y_keys=["y"], prepend_x=[0.0], prepend_y=[0.0]
            ),
            batch_shape=torch.Size(),
            active_dims=[0, 1],
            features_to_idx_mapper=None,
        )


def test_map_HammingDistanceKernel_to_categorical_without_ard():
    k_mapped = kernels.map(
        HammingDistanceKernel(
            ard=False,
        ),
        batch_shape=torch.Size(),
        active_dims=list(range(5)),
        features_to_idx_mapper=None,
    )

    assert isinstance(k_mapped, CategoricalKernel)
    assert k_mapped.active_dims.tolist() == [0, 1, 2, 3, 4]
    assert k_mapped.ard_num_dims is None
    assert k_mapped.lengthscale.shape == (1, 1)


def test_map_HammingDistanceKernel_to_categorical_with_ard():
    k_mapped = kernels.map(
        HammingDistanceKernel(
            ard=True,
        ),
        batch_shape=torch.Size(),
        active_dims=list(range(5)),
        features_to_idx_mapper=None,
    )

    assert isinstance(k_mapped, CategoricalKernel)
    assert k_mapped.active_dims.tolist() == [0, 1, 2, 3, 4]
    assert k_mapped.ard_num_dims == 5
    assert k_mapped.lengthscale.shape == (1, 5)


def test_map_IndexKernel():
    k_mapped = kernels.map(
        IndexKernel(
            num_categories=10,
            rank=3,
        ),
        batch_shape=torch.Size(),
        active_dims=list(range(5)),
        features_to_idx_mapper=None,
    )

    assert isinstance(k_mapped, GpytorchIndexKernel)
    assert k_mapped.active_dims.tolist() == [0, 1, 2, 3, 4]
    assert k_mapped.covar_factor.shape[0] == 10
    assert k_mapped.covar_factor.shape[1] == 3


def test_map_PositiveIndexKernel():
    k_mapped = kernels.map(
        PositiveIndexKernel(
            num_categories=8,
            rank=2,
        ),
        batch_shape=torch.Size(),
        active_dims=list(range(5)),
        features_to_idx_mapper=None,
    )

    assert isinstance(k_mapped, GpytorchIndexKernel)
    assert k_mapped.active_dims.tolist() == [0, 1, 2, 3, 4]
    assert k_mapped.covar_factor.shape[0] == 8
    assert k_mapped.covar_factor.shape[1] == 2


def test_map_multiple_kernels_on_feature_subsets():
    fmap = {
        "x_1": [0],
        "x_2": [1],
        "x_cat_1": [2],
        "x_cat_2": [3],
    }

    k_mapped = kernels.map(
        AdditiveKernel(
            kernels=[
                HammingDistanceKernel(
                    ard=True,
                    features=["x_cat_1", "x_cat_2"],
                ),
                RBFKernel(
                    features=["x_1", "x_2"],
                ),
            ]
        ),
        batch_shape=torch.Size(),
        active_dims=list(range(5)),
        features_to_idx_mapper=lambda ks: [i for k in ks for i in fmap[k]],
    )

    assert len(k_mapped.kernels) == 2

    assert isinstance(k_mapped.kernels[0], CategoricalKernel)
    assert k_mapped.kernels[0].active_dims.tolist() == [
        2,
        3,
    ]
    assert k_mapped.kernels[0].ard_num_dims == 2

    from gpytorch.kernels import RBFKernel as GpytorchRBFKernel

    assert isinstance(k_mapped.kernels[1], GpytorchRBFKernel)
    assert k_mapped.kernels[1].active_dims.tolist() == [0, 1]
    assert k_mapped.kernels[1].ard_num_dims == 2


def test_compute_active_dims_no_features_returns_active_dims():
    assert _compute_active_dims(
        data_model=FeatureSpecificKernel(
            type="test",
            features=None,
        ),
        active_dims=[1, 2, 3],
        features_to_idx_mapper=None,
    ) == [1, 2, 3]


def test_compute_active_dims_features_override_active_dims():
    assert _compute_active_dims(
        data_model=FeatureSpecificKernel(type="test", features=["x1", "x2"]),
        active_dims=[1, 2, 3],
        features_to_idx_mapper=lambda ks: [
            i for k in ks for i in {"x1": [4], "x2": [7]}[k]
        ],
    ) == [4, 7]


def test_compute_active_dims_fails_with_features_without_mapper():
    with pytest.raises(
        RuntimeError,
        match="features_to_idx_mapper must be defined when using only a subset of features",
    ):
        _compute_active_dims(
            data_model=FeatureSpecificKernel(type="test", features=["x1", "x2"]),
            active_dims=[1, 2, 3],
            features_to_idx_mapper=None,
        )


def test_map_PolynomialFeatureInteractionKernel():
    k = kernels.map(
        PolynomialFeatureInteractionKernel(
            kernels=[
                RBFKernel(features=["x1", "x2"]),
                MaternKernel(features=["x2", "x3"]),
            ],
            max_degree=2,
            include_self_interactions=False,
            outputscale_prior=THREESIX_SCALE_PRIOR(),
        ),
        active_dims=[],
        batch_shape=torch.Size(),
        features_to_idx_mapper=lambda ks: [int(k[1:]) for k in ks],
    )

    assert isinstance(k, aggregationKernels.PolynomialFeatureInteractionKernel)
    assert k.indices == [(0,), (1,), (0, 1)]
    assert k.outputscale.shape == (4,)

    assert isinstance(k.kernels[0], gpytorch.kernels.RBFKernel)
    assert k.kernels[0].active_dims.tolist() == [1, 2]

    assert isinstance(k.kernels[1], gpytorch.kernels.MaternKernel)
    assert k.kernels[1].active_dims.tolist() == [2, 3]


def test_map_WedgeKernel():
    feats = ["x1", "x2", "indicator"]
    conditions = [("x2", "indicator", ThresholdCondition(threshold=0.0, operator=">"))]

    # test dropping indicator feature
    data_model = WedgeKernel(
        base_kernel=RBFKernel(features=["x1", "x2"]),
        conditions=conditions,
    )

    k = kernels.map(
        data_model,
        active_dims=[0, 1, 2],
        batch_shape=torch.Size(),
        features_to_idx_mapper=lambda ks: list(map(feats.index, ks)),
    )
    assert isinstance(k, conditionalKernels.WedgeKernel)
    # the base kernel drops the indicator, and adds a second dimension for x2.
    assert k.base_kernel.active_dims.tolist() == [0, 1, 4]

    # test keeping indicator feature
    data_model = WedgeKernel(
        base_kernel=RBFKernel(features=["x1", "x2", "indicator"]),
        conditions=conditions,
    )

    k = kernels.map(
        data_model,
        active_dims=[0, 1, 2],
        batch_shape=torch.Size(),
        features_to_idx_mapper=lambda ks: list(map(feats.index, ks)),
    )

    assert isinstance(k, conditionalKernels.WedgeKernel)
    assert k.base_kernel.active_dims.tolist() == [0, 1, 2, 4]

    feats_dummy = ["dummy"] + feats
    data_model = WedgeKernel(
        base_kernel=RBFKernel(features=["x1", "x2"]), conditions=conditions, ard=True
    )

    k = kernels.map(
        data_model,
        active_dims=[1, 2, 3],
        batch_shape=torch.Size(),
        features_to_idx_mapper=lambda ks: list(map(feats_dummy.index, ks)),
    )
    k_sliced = kernels.map(
        data_model,
        active_dims=[0, 1, 2],
        batch_shape=torch.Size(),
        features_to_idx_mapper=lambda ks: list(map(feats.index, ks)),
    )

    assert k.ard_num_dims == 3
    X = torch.rand(6, 4)
    # the task column must not affect the covariance
    assert torch.allclose(k(X).to_dense(), k_sliced(X[:, 1:]).to_dense())


def test_map_spherical_linear_kernel():
    kernel = SphericalLinearKernel(ard=True)
    k = kernels.map(
        kernel,
        batch_shape=torch.Size(),
        active_dims=list(range(5)),
        features_to_idx_mapper=None,
    )
    assert isinstance(k, EQUIVALENTS[kernel.__class__])
    assert k.ard_num_dims == 5
    # Test lengthscale prior if configured
    kernel_with_prior = SphericalLinearKernel(
        ard=True, lengthscale_prior=GammaPrior(concentration=2.0, rate=0.15)
    )
    k_with_prior = kernels.map(
        kernel_with_prior,
        batch_shape=torch.Size(),
        active_dims=list(range(3)),
        features_to_idx_mapper=None,
    )
    assert hasattr(k_with_prior, "lengthscale_prior")


def test_spherical_linear_kernel_bounds():
    """Test SphericalLinearKernel with bounds (-1, 1) to verify buffer values."""
    input_dim = 5
    kernel = SphericalLinearKernel(ard=True, bounds=(-1.0, 1.0))
    k = kernels.map(
        kernel,
        batch_shape=torch.Size(),
        active_dims=list(range(input_dim)),
        features_to_idx_mapper=None,
    )
    # Check that _mins is a tensor of -1 with size input_dim
    assert torch.allclose(k.get_buffer("_mins"), torch.tensor([-1.0] * input_dim))
    assert k.get_buffer("_mins").shape == torch.Size([input_dim])

    # Check that _maxs is a tensor of 1 with size input_dim
    assert torch.allclose(k.get_buffer("_maxs"), torch.tensor([1.0] * input_dim))
    assert k.get_buffer("_maxs").shape == torch.Size([input_dim])

    # Check that _centers is a tensor of 0 with size input_dim
    assert torch.allclose(k.get_buffer("_centers"), torch.tensor([0.0] * input_dim))
    assert k.get_buffer("_centers").shape == torch.Size([input_dim])
