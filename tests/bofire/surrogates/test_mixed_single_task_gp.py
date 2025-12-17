"""
Integration tests for MixedSingleTaskGPSurrogate with kernel_dict feature.
Tests the surrogate mapper and full end-to-end functionality.
"""

from bofire.data_models.domain.api import Inputs, Outputs
from bofire.data_models.features.api import (
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
)
from bofire.data_models.kernels.api import (
    AdditiveKernel,
    HammingDistanceKernel,
    IndexKernel,
    MaternKernel,
    MultiplicativeKernel,
    RBFKernel,
    ScaleKernel,
)
from bofire.data_models.surrogates.api import MixedSingleTaskGPSurrogate
from bofire.surrogates.mapper import map_MixedSingleTaskGPSurrogate


"""Test suite for mapping MixedSingleTaskGPSurrogate with kernel_dict."""


def test_mapper_with_kernel_dict_none():
    """Test that mapper works correctly when kernel_dict is None."""
    data_model = MixedSingleTaskGPSurrogate(
        inputs=Inputs(
            features=[
                ContinuousInput(key="x1", bounds=[0, 1]),
                CategoricalInput(key="cat1", categories=["a", "b"]),
            ]
        ),
        outputs=Outputs(features=[ContinuousOutput(key="y")]),
        kernel_dict=None,
    )

    single_task_model = map_MixedSingleTaskGPSurrogate(data_model)

    # Should create an AdditiveKernel with sum and product kernels
    assert isinstance(single_task_model.kernel, AdditiveKernel)
    assert len(single_task_model.kernel.kernels) == 2

    # First kernel should be sum (additive)
    assert isinstance(single_task_model.kernel.kernels[0], ScaleKernel)
    assert isinstance(single_task_model.kernel.kernels[0].base_kernel, AdditiveKernel)

    # Second kernel should be product (multiplicative)
    assert isinstance(single_task_model.kernel.kernels[1], ScaleKernel)
    assert isinstance(
        single_task_model.kernel.kernels[1].base_kernel, MultiplicativeKernel
    )


def test_mapper_with_kernel_dict_continuous_only():
    """Test mapper with kernel_dict containing only continuous kernels."""
    data_model = MixedSingleTaskGPSurrogate(
        inputs=Inputs(
            features=[
                ContinuousInput(key="x1", bounds=[0, 1]),
                ContinuousInput(key="x2", bounds=[0, 1]),
                CategoricalInput(key="cat1", categories=["a", "b"]),
            ]
        ),
        outputs=Outputs(features=[ContinuousOutput(key="y")]),
        kernel_dict={
            "x1": RBFKernel(ard=True),
            "x2": MaternKernel(ard=True, nu=2.5),
            "cat1": HammingDistanceKernel(ard=True),
        },
    )

    single_task_model = map_MixedSingleTaskGPSurrogate(data_model)

    # Should still create an AdditiveKernel structure
    assert isinstance(single_task_model.kernel, AdditiveKernel)
    assert len(single_task_model.kernel.kernels) == 2


def test_mapper_with_kernel_dict_index_kernel():
    """Test mapper with kernel_dict containing IndexKernel."""
    data_model = MixedSingleTaskGPSurrogate(
        inputs=Inputs(
            features=[
                ContinuousInput(key="x1", bounds=[0, 1]),
                CategoricalInput(key="cat1", categories=["a", "b", "c"]),
            ]
        ),
        outputs=Outputs(features=[ContinuousOutput(key="y")]),
        kernel_dict={
            "x1": RBFKernel(ard=True),
            "cat1": IndexKernel(num_categories=3, rank=2),
        },
    )

    single_task_model = map_MixedSingleTaskGPSurrogate(data_model)

    # Verify the structure is created correctly
    assert isinstance(single_task_model.kernel, AdditiveKernel)

    # Check that features are assigned to kernels
    sum_kernel = single_task_model.kernel.kernels[0]
    assert isinstance(sum_kernel, ScaleKernel)
    assert isinstance(sum_kernel.base_kernel, AdditiveKernel)


def test_mapper_with_kernel_dict_multiple_features():
    """Test mapper with kernel_dict containing multiple features."""
    data_model = MixedSingleTaskGPSurrogate(
        inputs=Inputs(
            features=[
                ContinuousInput(key="x1", bounds=[0, 1]),
                ContinuousInput(key="x2", bounds=[0, 1]),
                CategoricalInput(key="cat1", categories=["a", "b"]),
                CategoricalInput(key="cat2", categories=["x", "y", "z"]),
            ]
        ),
        outputs=Outputs(features=[ContinuousOutput(key="y")]),
        kernel_dict={
            "x1": RBFKernel(ard=True),
            "x2": MaternKernel(ard=True, nu=2.5),
            "cat1": HammingDistanceKernel(ard=True),
            "cat2": IndexKernel(num_categories=3, rank=2),
        },
    )

    single_task_model = map_MixedSingleTaskGPSurrogate(data_model)

    # Should create proper kernel structure
    assert isinstance(single_task_model.kernel, AdditiveKernel)

    # Verify that all features are included
    sum_kernel = single_task_model.kernel.kernels[0]
    product_kernel = single_task_model.kernel.kernels[1]

    assert isinstance(sum_kernel, ScaleKernel)
    assert isinstance(product_kernel, ScaleKernel)


def test_mapper_purely_categorical():
    """Test mapper when model has no continuous features."""
    data_model = MixedSingleTaskGPSurrogate(
        inputs=Inputs(
            features=[
                CategoricalInput(key="cat1", categories=["a", "b"]),
                CategoricalInput(key="cat2", categories=["x", "y"]),
            ]
        ),
        outputs=Outputs(features=[ContinuousOutput(key="y")]),
    )

    single_task_model = map_MixedSingleTaskGPSurrogate(data_model)

    # Should create a simple ScaleKernel with categorical kernel
    assert isinstance(single_task_model.kernel, ScaleKernel)
    # Base kernel should be the categorical kernel
    assert single_task_model.kernel.base_kernel is not None


def test_mapper_feature_assignment_in_kernel_dict():
    """Test that mapper correctly assigns features to each kernel in kernel_dict."""
    data_model = MixedSingleTaskGPSurrogate(
        inputs=Inputs(
            features=[
                ContinuousInput(key="x1", bounds=[0, 1]),
                CategoricalInput(key="cat1", categories=["a", "b"]),
            ]
        ),
        outputs=Outputs(features=[ContinuousOutput(key="y")]),
        kernel_dict={
            "x1": RBFKernel(ard=True),
            "cat1": HammingDistanceKernel(ard=True),
        },
    )

    single_task_model = map_MixedSingleTaskGPSurrogate(data_model)

    # After mapping, the kernels should have features assigned
    assert isinstance(single_task_model.kernel, AdditiveKernel)

    # Verify structure
    sum_kernel = single_task_model.kernel.kernels[0]
    assert isinstance(sum_kernel.base_kernel, AdditiveKernel)

    # Check that kernels have features assigned
    for kernel in sum_kernel.base_kernel.kernels:
        if isinstance(kernel, ScaleKernel):
            # Categorical kernels are wrapped in ScaleKernel
            assert kernel.base_kernel.features is not None
        else:
            # Continuous kernels are added directly
            assert kernel.features is not None
