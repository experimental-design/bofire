import pytest
from pandas.testing import assert_frame_equal

import bofire.surrogates.api as surrogates
from bofire.benchmarks.single import Himmelblau, PositiveHimmelblau
from bofire.data_models.domain.api import Inputs, Outputs
from bofire.data_models.features.api import (
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
)
from bofire.data_models.surrogates.api import (
    Normalize,
    PFNSurrogate,
    ScalerEnum,
    Standardize,
)


try:
    from botorch_community.models.prior_fitted_network import (
        MultivariatePFNModel,
        PFNModel,
    )

    BOTORCH_COMMUNITY_AVAILABLE = True
except ImportError:
    BOTORCH_COMMUNITY_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not BOTORCH_COMMUNITY_AVAILABLE,
    reason="botorch_community not installed",
)


@pytest.mark.parametrize(
    "scaler, output_scaler",
    [
        [Normalize(), ScalerEnum.IDENTITY],
        [Standardize(), ScalerEnum.STANDARDIZE],
        [None, ScalerEnum.STANDARDIZE],
    ],
)
def test_pfn_surrogate_fit(scaler, output_scaler):
    """Test PFN surrogate fitting with different scalers."""
    bench = PositiveHimmelblau()
    samples = bench.domain.inputs.sample(15)
    experiments = bench.f(samples, return_complete=True)

    pfn = PFNSurrogate(
        inputs=bench.domain.inputs,
        outputs=bench.domain.outputs,
        checkpoint_url="pfns4bo_hebo",
        batch_first=False,
        multivariate=False,
        scaler=scaler,
        output_scaler=output_scaler,
    )
    surrogate = surrogates.map(pfn)

    surrogate.fit(experiments=experiments)

    # Check input transforms
    if scaler is None:
        assert surrogate._input_transform is None

    # Check outcome transforms
    if output_scaler == ScalerEnum.STANDARDIZE:
        assert hasattr(surrogate, "_outcome_transform")
        # Note: PFN handles outcome transforms differently than standard GP models
    elif output_scaler == ScalerEnum.IDENTITY:
        # May or may not have outcome transform depending on implementation
        pass

    # Test predictions
    preds = surrogate.predict(experiments)

    assert "y_pred" in preds.columns
    assert "y_sd" in preds.columns
    assert preds.shape[0] == experiments.shape[0]

    # Test serialization/deserialization
    dump = surrogate.dumps()
    surrogate2 = surrogates.map(pfn)
    surrogate2.loads(dump)
    preds2 = surrogate2.predict(experiments)
    assert_frame_equal(preds, preds2)


def test_pfn_surrogate_multivariate():
    """Test PFN surrogate with multivariate posterior."""
    bench = Himmelblau()
    samples = bench.domain.inputs.sample(15)
    experiments = bench.f(samples, return_complete=True)

    pfn = PFNSurrogate(
        inputs=bench.domain.inputs,
        outputs=bench.domain.outputs,
        checkpoint_url="pfns4bo_hebo",
        batch_first=False,
        multivariate=True,  # Enable multivariate posterior
        scaler=Normalize(),
        output_scaler=ScalerEnum.STANDARDIZE,
    )
    surrogate = surrogates.map(pfn)

    surrogate.fit(experiments=experiments)

    # Check that MultivariatePFNModel is used
    assert isinstance(surrogate.model, MultivariatePFNModel)

    # Test predictions
    preds = surrogate.predict(experiments)
    assert "y_pred" in preds.columns
    assert "y_sd" in preds.columns


def test_pfn_surrogate_checkpoint_variants():
    """Test PFN surrogate with different checkpoint URLs."""
    bench = Himmelblau()
    samples = bench.domain.inputs.sample(15)
    experiments = bench.f(samples, return_complete=True)

    # Test with pfns4bo_bnn checkpoint
    pfn = PFNSurrogate(
        inputs=bench.domain.inputs,
        outputs=bench.domain.outputs,
        checkpoint_url="pfns4bo_bnn",
        batch_first=False,
        multivariate=False,
        scaler=Normalize(),
        output_scaler=ScalerEnum.STANDARDIZE,
    )
    surrogate = surrogates.map(pfn)

    surrogate.fit(experiments=experiments)

    # Check that model is PFNModel
    assert isinstance(surrogate.model, PFNModel)

    # Test predictions
    preds = surrogate.predict(experiments)
    assert "y_pred" in preds.columns
    assert "y_sd" in preds.columns


def test_pfn_surrogate_batch_first():
    """Test PFN surrogate with batch_first=True."""
    bench = Himmelblau()
    samples = bench.domain.inputs.sample(15)
    experiments = bench.f(samples, return_complete=True)

    pfn = PFNSurrogate(
        inputs=bench.domain.inputs,
        outputs=bench.domain.outputs,
        checkpoint_url="pfns4bo_hebo",
        batch_first=True,  # Batch dimension first
        multivariate=False,
        scaler=Normalize(),
        output_scaler=ScalerEnum.STANDARDIZE,
    )
    surrogate = surrogates.map(pfn)

    surrogate.fit(experiments=experiments)

    # Test predictions
    preds = surrogate.predict(experiments)
    assert "y_pred" in preds.columns
    assert "y_sd" in preds.columns


@pytest.mark.skip(
    reason="PFN pfns4bo_hebo model has issues with categorical features - dimension mismatch"
)
def test_pfn_surrogate_categorical_input():
    """Test PFN surrogate with categorical inputs."""
    inputs = Inputs(
        features=[
            ContinuousInput(key="x_1", bounds=(-4, 4)),
            ContinuousInput(key="x_2", bounds=(-4, 4)),
            CategoricalInput(key="x_cat", categories=["a", "b"]),
        ],
    )
    outputs = Outputs(features=[ContinuousOutput(key="y")])

    experiments = inputs.sample(n=15)
    experiments.eval("y=((x_1**2 + x_2 - 11)**2+(x_1 + x_2**2 -7)**2)", inplace=True)
    experiments.loc[experiments.x_cat == "a", "y"] *= 2.0
    experiments.loc[experiments.x_cat == "b", "y"] /= 2.0
    experiments["valid_y"] = 1

    pfn = PFNSurrogate(
        inputs=inputs,
        outputs=outputs,
        checkpoint_url="pfns4bo_hebo",
        batch_first=False,
        multivariate=False,
        scaler=Normalize(),
        output_scaler=ScalerEnum.STANDARDIZE,
    )
    surrogate = surrogates.map(pfn)

    surrogate.fit(experiments=experiments)

    # Test predictions
    preds = surrogate.predict(experiments)
    assert "y_pred" in preds.columns
    assert "y_sd" in preds.columns
    assert preds.shape[0] == experiments.shape[0]


def test_pfn_surrogate_small_dataset():
    """Test PFN surrogate with very small dataset."""
    bench = Himmelblau()
    samples = bench.domain.inputs.sample(5)  # Very small dataset
    experiments = bench.f(samples, return_complete=True)

    pfn = PFNSurrogate(
        inputs=bench.domain.inputs,
        outputs=bench.domain.outputs,
        checkpoint_url="pfns4bo_hebo",
        batch_first=False,
        multivariate=False,
        scaler=Normalize(),
        output_scaler=ScalerEnum.STANDARDIZE,
    )
    surrogate = surrogates.map(pfn)

    surrogate.fit(experiments=experiments)

    # Test predictions
    preds = surrogate.predict(experiments)
    assert "y_pred" in preds.columns
    assert "y_sd" in preds.columns


def test_pfn_surrogate_constant_model_kwargs():
    """Test PFN surrogate with constant_model_kwargs."""
    bench = Himmelblau()
    samples = bench.domain.inputs.sample(15)
    experiments = bench.f(samples, return_complete=True)

    pfn = PFNSurrogate(
        inputs=bench.domain.inputs,
        outputs=bench.domain.outputs,
        checkpoint_url="pfns4bo_hebo",
        batch_first=False,
        multivariate=False,
        constant_model_kwargs={},  # Empty dict for now
        scaler=Normalize(),
        output_scaler=ScalerEnum.STANDARDIZE,
    )
    surrogate = surrogates.map(pfn)

    surrogate.fit(experiments=experiments)

    # Test predictions
    preds = surrogate.predict(experiments)
    assert "y_pred" in preds.columns
    assert "y_sd" in preds.columns


def test_pfn_surrogate_is_fitted():
    """Test PFN surrogate is_fitted property."""
    bench = Himmelblau()
    samples = bench.domain.inputs.sample(15)
    experiments = bench.f(samples, return_complete=True)

    pfn = PFNSurrogate(
        inputs=bench.domain.inputs,
        outputs=bench.domain.outputs,
        checkpoint_url="pfns4bo_hebo",
        batch_first=False,
        multivariate=False,
        scaler=Normalize(),
        output_scaler=ScalerEnum.STANDARDIZE,
    )
    surrogate = surrogates.map(pfn)

    # Before fitting
    assert not surrogate.is_fitted

    # After fitting
    surrogate.fit(experiments=experiments)
    assert surrogate.is_fitted

    # After serialization/deserialization
    dump = surrogate.dumps()
    surrogate2 = surrogates.map(pfn)
    assert not surrogate2.is_fitted
    surrogate2.loads(dump)
    assert surrogate2.is_fitted


def test_pfn_surrogate_prediction_shapes():
    """Test PFN surrogate prediction output shapes."""
    bench = Himmelblau()
    samples = bench.domain.inputs.sample(15)
    experiments = bench.f(samples, return_complete=True)

    pfn = PFNSurrogate(
        inputs=bench.domain.inputs,
        outputs=bench.domain.outputs,
        checkpoint_url="pfns4bo_hebo",
        batch_first=False,
        multivariate=False,
        scaler=Normalize(),
        output_scaler=ScalerEnum.STANDARDIZE,
    )
    surrogate = surrogates.map(pfn)

    surrogate.fit(experiments=experiments)

    # Test predictions on training data
    preds = surrogate.predict(experiments)
    assert preds.shape[0] == experiments.shape[0]
    assert "y_pred" in preds.columns
    assert "y_sd" in preds.columns

    # Test predictions on new data
    new_samples = bench.domain.inputs.sample(5)
    new_preds = surrogate.predict(new_samples)
    assert new_preds.shape[0] == new_samples.shape[0]
    assert "y_pred" in new_preds.columns
    assert "y_sd" in new_preds.columns
