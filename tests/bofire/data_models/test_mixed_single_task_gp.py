"""
Comprehensive tests for MixedSingleTaskGPSurrogate with kernel_dict feature.
Tests all validation logic and edge cases for the new kernel_dict functionality.
"""

import pytest

from bofire.data_models.domain.api import Inputs, Outputs
from bofire.data_models.features.api import (
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
)
from bofire.data_models.kernels.api import (
    HammingDistanceKernel,
    IndexKernel,
    MaternKernel,
    PositiveIndexKernel,
    RBFKernel,
)
from bofire.data_models.surrogates.api import MixedSingleTaskGPSurrogate


"""Test suite for kernel_dict validation in MixedSingleTaskGPSurrogate."""


def test_kernel_dict_with_matching_features():
    """Test that kernel_dict works when all features are present."""
    surrogate = MixedSingleTaskGPSurrogate(
        inputs=Inputs(
            features=[
                ContinuousInput(key="x1", bounds=[0, 1]),
                ContinuousInput(key="x2", bounds=[0, 1]),
                CategoricalInput(key="cat1", categories=["a", "b", "c"]),
            ]
        ),
        outputs=Outputs(features=[ContinuousOutput(key="y")]),
        kernel_dict={
            "x1": RBFKernel(ard=True),
            "x2": MaternKernel(ard=True, nu=2.5),
            "cat1": HammingDistanceKernel(ard=True),
        },
    )
    assert surrogate.kernel_dict is not None
    assert len(surrogate.kernel_dict) == 3


def test_kernel_dict_with_index_kernel():
    """Test that kernel_dict works with IndexKernel for categorical features."""
    surrogate = MixedSingleTaskGPSurrogate(
        inputs=Inputs(
            features=[
                ContinuousInput(key="x1", bounds=[0, 1]),
                CategoricalInput(key="cat1", categories=["a", "b", "c", "d"]),
            ]
        ),
        outputs=Outputs(features=[ContinuousOutput(key="y")]),
        kernel_dict={
            "x1": RBFKernel(ard=True),
            "cat1": IndexKernel(num_categories=4, rank=2),
        },
    )
    assert surrogate.kernel_dict is not None
    assert isinstance(surrogate.kernel_dict["cat1"], IndexKernel)


def test_kernel_dict_none_uses_defaults():
    """Test that None kernel_dict uses default kernels."""
    surrogate = MixedSingleTaskGPSurrogate(
        inputs=Inputs(
            features=[
                ContinuousInput(key="x1", bounds=[0, 1]),
                CategoricalInput(key="cat1", categories=["a", "b"]),
            ]
        ),
        outputs=Outputs(features=[ContinuousOutput(key="y")]),
        kernel_dict=None,
    )
    assert surrogate.kernel_dict is None
    assert surrogate.continuous_kernel is not None
    assert surrogate.categorical_kernel is not None


def test_kernel_dict_length_mismatch_raises_error():
    """Test that ValidationError is raised when kernel_dict length doesn't match features."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="must match the number of all features"):
        MixedSingleTaskGPSurrogate(
            inputs=Inputs(
                features=[
                    ContinuousInput(key="x1", bounds=[0, 1]),
                    CategoricalInput(key="cat1", categories=["a", "b"]),
                ]
            ),
            outputs=Outputs(features=[ContinuousOutput(key="y")]),
            kernel_dict={
                "x1": RBFKernel(ard=True),
                # Missing cat1 - should fail
            },
        )


def test_kernel_dict_unknown_feature_raises_error():
    """Test that ValueError is raised when kernel_dict contains unknown feature."""
    from pydantic import ValidationError

    # This will first fail on length mismatch (3 != 2)
    with pytest.raises(ValidationError, match="must match the number of all features"):
        MixedSingleTaskGPSurrogate(
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
                "x_unknown": RBFKernel(ard=True),  # Unknown feature
            },
        )


def test_kernel_dict_wrong_key_with_correct_length_raises_error():
    """Test that ValueError is raised when kernel_dict key doesn't exist in inputs (but length matches)."""
    with pytest.raises(ValueError, match="not present in the inputs"):
        MixedSingleTaskGPSurrogate(
            inputs=Inputs(
                features=[
                    ContinuousInput(key="x1", bounds=[0, 1]),
                    CategoricalInput(key="cat1", categories=["a", "b"]),
                ]
            ),
            outputs=Outputs(features=[ContinuousOutput(key="y")]),
            kernel_dict={
                "x_unknown": RBFKernel(ard=True),  # Wrong key
                "cat1": HammingDistanceKernel(ard=True),
            },
        )


def test_kernel_dict_categorical_with_continuous_kernel_raises_error():
    """Test that ValueError is raised when categorical feature mapped to continuous kernel."""
    with pytest.raises(
        ValueError, match="categorical and must be mapped to a categorical kernel"
    ):
        MixedSingleTaskGPSurrogate(
            inputs=Inputs(
                features=[
                    ContinuousInput(key="x1", bounds=[0, 1]),
                    CategoricalInput(key="cat1", categories=["a", "b"]),
                ]
            ),
            outputs=Outputs(features=[ContinuousOutput(key="y")]),
            kernel_dict={
                "x1": RBFKernel(ard=True),
                "cat1": RBFKernel(ard=True),  # Wrong kernel type
            },
        )


def test_kernel_dict_continuous_with_categorical_kernel_raises_error():
    """Test that ValueError is raised when continuous feature mapped to categorical kernel."""
    with pytest.raises(
        ValueError, match="continuous and must be mapped to a continuous kernel"
    ):
        MixedSingleTaskGPSurrogate(
            inputs=Inputs(
                features=[
                    ContinuousInput(key="x1", bounds=[0, 1]),
                    CategoricalInput(key="cat1", categories=["a", "b"]),
                ]
            ),
            outputs=Outputs(features=[ContinuousOutput(key="y")]),
            kernel_dict={
                "x1": HammingDistanceKernel(ard=True),  # Wrong kernel type
                "cat1": HammingDistanceKernel(ard=True),
            },
        )


def test_kernel_dict_index_kernel_num_categories_mismatch_raises_error():
    """Test that ValueError is raised when IndexKernel num_categories doesn't match input categories."""
    with pytest.raises(ValueError, match="num_categories.*does not match"):
        MixedSingleTaskGPSurrogate(
            inputs=Inputs(
                features=[
                    ContinuousInput(key="x1", bounds=[0, 1]),
                    CategoricalInput(
                        key="cat1", categories=["a", "b", "c"]
                    ),  # 3 categories
                ]
            ),
            outputs=Outputs(features=[ContinuousOutput(key="y")]),
            kernel_dict={
                "x1": RBFKernel(ard=True),
                "cat1": IndexKernel(num_categories=5, rank=2),  # Mismatch: 5 != 3
            },
        )


def test_kernel_dict_index_kernel_correct_num_categories():
    """Test that IndexKernel with correct num_categories works."""
    surrogate = MixedSingleTaskGPSurrogate(
        inputs=Inputs(
            features=[
                ContinuousInput(key="x1", bounds=[0, 1]),
                CategoricalInput(
                    key="cat1", categories=["a", "b", "c"]
                ),  # 3 categories
            ]
        ),
        outputs=Outputs(features=[ContinuousOutput(key="y")]),
        kernel_dict={
            "x1": RBFKernel(ard=True),
            "cat1": IndexKernel(num_categories=3, rank=2),  # Correct: 3 == 3
        },
    )
    assert surrogate.kernel_dict["cat1"].num_categories == 3


def test_kernel_dict_multiple_categorical_features():
    """Test kernel_dict with multiple categorical features."""
    surrogate = MixedSingleTaskGPSurrogate(
        inputs=Inputs(
            features=[
                ContinuousInput(key="x1", bounds=[0, 1]),
                CategoricalInput(key="cat1", categories=["a", "b"]),
                CategoricalInput(key="cat2", categories=["x", "y", "z"]),
            ]
        ),
        outputs=Outputs(features=[ContinuousOutput(key="y")]),
        kernel_dict={
            "x1": MaternKernel(ard=True, nu=2.5),
            "cat1": IndexKernel(num_categories=2, rank=1),
            "cat2": HammingDistanceKernel(ard=True),
        },
    )
    assert len(surrogate.kernel_dict) == 3
    assert isinstance(surrogate.kernel_dict["cat1"], IndexKernel)
    assert isinstance(surrogate.kernel_dict["cat2"], HammingDistanceKernel)


def test_kernel_dict_multiple_continuous_features():
    """Test kernel_dict with multiple continuous features using different kernels."""
    surrogate = MixedSingleTaskGPSurrogate(
        inputs=Inputs(
            features=[
                ContinuousInput(key="x1", bounds=[0, 1]),
                ContinuousInput(key="x2", bounds=[0, 1]),
                ContinuousInput(key="x3", bounds=[0, 1]),
                CategoricalInput(key="cat1", categories=["a", "b"]),
            ]
        ),
        outputs=Outputs(features=[ContinuousOutput(key="y")]),
        kernel_dict={
            "x1": RBFKernel(ard=True),
            "x2": MaternKernel(ard=True, nu=2.5),
            "x3": MaternKernel(ard=True, nu=1.5),
            "cat1": PositiveIndexKernel(num_categories=2, rank=1),
        },
    )
    assert len(surrogate.kernel_dict) == 4
    assert isinstance(surrogate.kernel_dict["x1"], RBFKernel)
    assert isinstance(surrogate.kernel_dict["x2"], MaternKernel)
    assert surrogate.kernel_dict["x2"].nu == 2.5
    assert surrogate.kernel_dict["x3"].nu == 1.5
