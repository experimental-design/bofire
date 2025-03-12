import gpytorch
import gpytorch.kernels
import pytest
import torch
from botorch.models.kernels import InfiniteWidthBNNKernel as BNNKernel
from botorch.models.kernels.categorical import CategoricalKernel

import bofire
import bofire.kernels.aggregation as aggregationKernels
import bofire.kernels.api as kernels
import bofire.kernels.shape as shapeKernels
from bofire.data_models.kernels.api import (
    AdditiveKernel,
    FeatureSpecificKernel,
    HammingDistanceKernel,
    InfiniteWidthBNNKernel,
    LinearKernel,
    MaternKernel,
    MultiplicativeKernel,
    PolynomialFeatureInteractionKernel,
    PolynomialKernel,
    RBFKernel,
    ScaleKernel,
    TanimotoKernel,
    WassersteinKernel,
)
from bofire.data_models.priors.api import THREESIX_SCALE_PRIOR, GammaPrior
from bofire.kernels.categorical import HammingKernelWithOneHots
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
    WassersteinKernel: shapeKernels.WassersteinKernel,
    InfiniteWidthBNNKernel: BNNKernel,
    PolynomialFeatureInteractionKernel: aggregationKernels.PolynomialFeatureInteractionKernel,
}


def test_map(kernel_spec: Spec):
    kernel = kernel_spec.cls(**kernel_spec.typed_spec())
    if isinstance(kernel, HammingDistanceKernel) and kernel.features is not None:
        return
    gkernel = kernels.map(
        kernel,
        batch_shape=torch.Size(),
        ard_num_dims=10,
        active_dims=list(range(5)),
        features_to_idx_mapper=None,
    )
    assert isinstance(gkernel, EQUIVALENTS[kernel.__class__])


def test_map_scale_kernel():
    kernel = ScaleKernel(
        base_kernel=RBFKernel(), outputscale_prior=THREESIX_SCALE_PRIOR()
    )
    k = kernels.map(
        kernel,
        batch_shape=torch.Size(),
        ard_num_dims=10,
        active_dims=list(range(5)),
        features_to_idx_mapper=None,
    )
    assert hasattr(k, "outputscale_prior")
    assert isinstance(k.outputscale_prior, gpytorch.priors.GammaPrior)
    kernel = ScaleKernel(base_kernel=RBFKernel())
    k = kernels.map(
        kernel,
        batch_shape=torch.Size(),
        ard_num_dims=10,
        active_dims=list(range(5)),
        features_to_idx_mapper=None,
    )
    assert hasattr(k, "outputscale_prior") is False


def test_map_polynomial_kernel():
    kernel = PolynomialKernel(power=2, offset_prior=THREESIX_SCALE_PRIOR())
    k = kernels.map(
        kernel,
        batch_shape=torch.Size(),
        ard_num_dims=10,
        active_dims=list(range(5)),
        features_to_idx_mapper=None,
    )
    assert hasattr(k, "offset_prior")
    assert isinstance(k.offset_prior, gpytorch.priors.GammaPrior)
    kernel = PolynomialKernel(power=2)
    k = kernels.map(
        kernel,
        batch_shape=torch.Size(),
        ard_num_dims=10,
        active_dims=list(range(5)),
        features_to_idx_mapper=None,
    )
    assert hasattr(k, "offset_prior") is False


@pytest.mark.parametrize(
    "kernel, ard_num_dims, active_dims, expected_kernel",
    [
        (
            RBFKernel(
                ard=False,
                lengthscale_prior=GammaPrior(concentration=2.0, rate=0.15),
            ),
            10,
            list(range(5)),
            gpytorch.kernels.RBFKernel,
        ),
        (
            RBFKernel(ard=False),
            10,
            list(range(5)),
            gpytorch.kernels.RBFKernel,
        ),
        (RBFKernel(ard=True), 10, list(range(5)), gpytorch.kernels.RBFKernel),
        (
            MaternKernel(
                ard=False,
                lengthscale_prior=GammaPrior(concentration=2.0, rate=0.15),
            ),
            10,
            list(range(5)),
            gpytorch.kernels.MaternKernel,
        ),
        (MaternKernel(ard=False), 10, list(range(5)), gpytorch.kernels.MaternKernel),
        (MaternKernel(ard=True), 10, list(range(5)), gpytorch.kernels.MaternKernel),
        (
            MaternKernel(ard=False, nu=2.5),
            10,
            list(range(5)),
            gpytorch.kernels.MaternKernel,
        ),
        (
            MaternKernel(ard=True, nu=1.5),
            10,
            list(range(5)),
            gpytorch.kernels.MaternKernel,
        ),
        (LinearKernel(), 10, list(range(5)), gpytorch.kernels.LinearKernel),
    ],
)
def test_map_continuous_kernel(kernel, ard_num_dims, active_dims, expected_kernel):
    k = kernels.map(
        kernel,
        batch_shape=torch.Size(),
        ard_num_dims=ard_num_dims,
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
    "kernel, ard_num_dims, active_dims, expected_kernel",
    [
        (
            TanimotoKernel(ard=False),
            10,
            list(range(5)),
            bofire.kernels.fingerprint_kernels.tanimoto_kernel.TanimotoKernel,
        ),
        (
            TanimotoKernel(ard=True),
            10,
            list(range(5)),
            bofire.kernels.fingerprint_kernels.tanimoto_kernel.TanimotoKernel,
        ),
    ],
)
def test_map_molecular_kernel(kernel, ard_num_dims, active_dims, expected_kernel):
    k = kernels.map(
        kernel,
        batch_shape=torch.Size(),
        ard_num_dims=ard_num_dims,
        active_dims=active_dims,
        features_to_idx_mapper=None,
    )
    assert isinstance(k, expected_kernel)

    if kernel.ard is False:
        assert k.ard_num_dims is None
    else:
        assert k.ard_num_dims == len(active_dims)
    assert torch.eq(k.active_dims, torch.tensor(active_dims, dtype=torch.int64)).all()


def test_map_wasserstein_kernel():
    kernel = WassersteinKernel(
        squared=False,
        lengthscale_prior=GammaPrior(concentration=2.0, rate=0.15),
    )
    k = kernels.map(
        kernel,
        batch_shape=torch.Size(),
        ard_num_dims=10,
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
        ard_num_dims=10,
        active_dims=list(range(5)),
        features_to_idx_mapper=None,
    )
    assert k.squared is True
    assert hasattr(k, "lengthscale_prior") is False


def test_map_HammingDistanceKernel_to_onehot_with_ard():
    fmap = {
        "x_cat_1": [5, 6, 7, 8],
        "x_cat_2": [2, 3],
    }

    k_mapped = kernels.map(
        HammingDistanceKernel(
            ard=True,
            features=["x_cat_1", "x_cat_2"],
        ),
        batch_shape=torch.Size(),
        ard_num_dims=10,
        active_dims=list(range(5)),
        features_to_idx_mapper=lambda ks: [i for k in ks for i in fmap[k]],
    )

    assert isinstance(k_mapped, HammingKernelWithOneHots)
    assert k_mapped.active_dims.tolist() == [5, 6, 7, 8, 2, 3]
    assert k_mapped.ard_num_dims == 6
    assert k_mapped.lengthscale.shape == (1, 6)
    assert k_mapped.trx.categorical_features == {0: 4, 4: 2}


def test_map_HammingDistanceKernel_to_onehot_without_ard():
    fmap = {
        "x_cat_1": [5, 6, 7, 8],
        "x_cat_2": [2, 3],
    }

    k_mapped = kernels.map(
        HammingDistanceKernel(
            ard=False,
            features=["x_cat_1", "x_cat_2"],
        ),
        batch_shape=torch.Size(),
        ard_num_dims=10,
        active_dims=list(range(5)),
        features_to_idx_mapper=lambda ks: [i for k in ks for i in fmap[k]],
    )

    assert isinstance(k_mapped, HammingKernelWithOneHots)
    assert k_mapped.active_dims.tolist() == [5, 6, 7, 8, 2, 3]
    assert k_mapped.ard_num_dims is None
    assert k_mapped.lengthscale.shape == (1, 1)
    assert k_mapped.trx.categorical_features == {0: 4, 4: 2}


def test_map_HammingDistanceKernel_to_categorical_without_ard():
    k_mapped = kernels.map(
        HammingDistanceKernel(
            ard=False,
        ),
        batch_shape=torch.Size(),
        ard_num_dims=10,
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
        ard_num_dims=10,
        active_dims=list(range(5)),
        features_to_idx_mapper=None,
    )

    assert isinstance(k_mapped, CategoricalKernel)
    assert k_mapped.active_dims.tolist() == [0, 1, 2, 3, 4]
    assert k_mapped.ard_num_dims == 5
    assert k_mapped.lengthscale.shape == (1, 5)


def test_map_HammingDistanceKernel_to_onehot_checks_dimension_overlap():
    fmap = {
        "x_cat_1": [3, 4],
        "x_cat_2": [2, 3],
    }

    with pytest.raises(
        RuntimeError,
        match=r"indices \[3\] are used in more than one categorical feature",
    ):
        kernels.map(
            HammingDistanceKernel(
                ard=True,
                features=["x_cat_1", "x_cat_2"],
            ),
            batch_shape=torch.Size(),
            ard_num_dims=10,
            active_dims=list(range(5)),
            features_to_idx_mapper=lambda ks: [i for k in ks for i in fmap[k]],
        )


def test_map_HammingDistanceKernel_to_onehot_checks_onehot_encoding():
    fmap = {
        "x_cat_1": [4],
        "x_cat_2": [2, 3],
    }

    with pytest.raises(
        RuntimeError,
        match="feature x_cat_1 is supposed to be one-hot encoded but is mapped to a single dimension",
    ):
        kernels.map(
            HammingDistanceKernel(
                ard=True,
                features=["x_cat_1", "x_cat_2"],
            ),
            batch_shape=torch.Size(),
            ard_num_dims=10,
            active_dims=list(range(5)),
            features_to_idx_mapper=lambda ks: [i for k in ks for i in fmap[k]],
        )


def test_map_multiple_kernels_on_feature_subsets():
    fmap = {
        "x_1": [0],
        "x_2": [1],
        "x_cat_1": [2, 3],
        "x_cat_2": [4, 5],
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
        ard_num_dims=10,
        active_dims=list(range(5)),
        features_to_idx_mapper=lambda ks: [i for k in ks for i in fmap[k]],
    )

    assert len(k_mapped.kernels) == 2

    assert isinstance(k_mapped.kernels[0], HammingKernelWithOneHots)
    assert k_mapped.kernels[0].active_dims.tolist() == [2, 3, 4, 5]
    assert k_mapped.kernels[0].ard_num_dims == 4

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
        ard_num_dims=1,
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
