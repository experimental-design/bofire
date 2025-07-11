import importlib

import pandas as pd
import pytest
import torch
from botorch.models import MixedSingleTaskGP, SingleTaskGP
from botorch.models.robust_relevance_pursuit_model import (
    RobustRelevancePursuitSingleTaskGP,
)
from botorch.models.transforms.input import (
    ChainedInputTransform,
    InputStandardize,
    Normalize,
    OneHotToNumeric,
)
from botorch.models.transforms.outcome import Standardize
from pandas.testing import assert_frame_equal
from pydantic import ValidationError

import bofire.surrogates.api as surrogates
from bofire.benchmarks.api import Hartmann, Himmelblau
from bofire.data_models.domain.api import Inputs, Outputs
from bofire.data_models.enum import CategoricalEncodingEnum, RegressionMetricsEnum
from bofire.data_models.features.api import (
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
    MolecularInput,
)
from bofire.data_models.kernels.api import (
    AdditiveKernel,
    HammingDistanceKernel,
    MaternKernel,
    RBFKernel,
    ScaleKernel,
    TanimotoKernel,
)
from bofire.data_models.molfeatures.api import MordredDescriptors
from bofire.data_models.priors.api import (
    HVARFNER_LENGTHSCALE_PRIOR,
    HVARFNER_NOISE_PRIOR,
    MBO_LENGTHCALE_PRIOR,
    MBO_NOISE_PRIOR,
    MBO_OUTPUTSCALE_PRIOR,
    ROBUSTGP_LENGTHSCALE_CONSTRAINT,
    THREESIX_LENGTHSCALE_PRIOR,
    THREESIX_NOISE_PRIOR,
    THREESIX_SCALE_PRIOR,
)
from bofire.data_models.surrogates.api import (
    MixedSingleTaskGPSurrogate,
    RobustSingleTaskGPSurrogate,
    ScalerEnum,
    SingleTaskGPHyperconfig,
    SingleTaskGPSurrogate,
)
from bofire.data_models.surrogates.trainable import metrics2objectives


RDKIT_AVAILABLE = importlib.util.find_spec("rdkit") is not None


@pytest.mark.parametrize(
    "kernel, scaler, output_scaler",
    [
        (
            ScaleKernel(base_kernel=RBFKernel(ard=True)),
            ScalerEnum.NORMALIZE,
            ScalerEnum.STANDARDIZE,
        ),
        (
            ScaleKernel(base_kernel=RBFKernel(ard=False)),
            ScalerEnum.STANDARDIZE,
            ScalerEnum.STANDARDIZE,
        ),
        (
            ScaleKernel(base_kernel=RBFKernel(ard=False)),
            ScalerEnum.IDENTITY,
            ScalerEnum.IDENTITY,
        ),
    ],
)
def test_SingleTaskGPModel(kernel, scaler, output_scaler):
    inputs = Inputs(
        features=[
            ContinuousInput(
                key=f"x_{i + 1}",
                bounds=(-4, 4),
            )
            for i in range(2)
        ],
    )
    outputs = Outputs(features=[ContinuousOutput(key="y")])
    experiments = inputs.sample(n=10)
    experiments.eval("y=((x_1**2 + x_2 - 11)**2+(x_1 + x_2**2 -7)**2)", inplace=True)
    experiments["valid_y"] = 1
    model = SingleTaskGPSurrogate(
        inputs=inputs,
        outputs=outputs,
        kernel=kernel,
        scaler=scaler,
        output_scaler=output_scaler,
    )
    model = surrogates.map(model)
    samples = inputs.sample(5)
    # test error on non fitted model
    with pytest.raises(ValueError):
        model.predict(samples)
    model.fit(experiments)
    # dump the model
    dump = model.dumps()
    # make predictions
    samples2 = samples.copy()
    samples2 = samples2.astype({"x_1": "object"})
    preds = model.predict(samples2)
    assert preds.shape == (5, 2)
    # check that model is composed correctly
    assert isinstance(model.model, SingleTaskGP)
    if output_scaler == ScalerEnum.STANDARDIZE:
        assert isinstance(model.model.outcome_transform, Standardize)
    elif output_scaler == ScalerEnum.IDENTITY:
        assert not hasattr(model.model, "outcome_transform")
    if scaler == ScalerEnum.NORMALIZE:
        assert isinstance(model.model.input_transform, Normalize)
    elif scaler == ScalerEnum.STANDARDIZE:
        assert isinstance(model.model.input_transform, InputStandardize)
    else:
        with pytest.raises(AttributeError):
            assert model.model.input_transform is None
    assert model.is_compatibilized is False
    # reload the model from dump and check for equality in predictions
    model2 = SingleTaskGPSurrogate(
        inputs=inputs,
        outputs=outputs,
        kernel=kernel,
        scaler=scaler,
    )
    model2 = surrogates.map(model2)
    model2.loads(dump)
    preds2 = model2.predict(samples)
    assert_frame_equal(preds, preds2)


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
@pytest.mark.parametrize(
    "kernel, scaler, output_scaler",
    [
        (
            ScaleKernel(base_kernel=RBFKernel(ard=True)),
            ScalerEnum.NORMALIZE,
            ScalerEnum.STANDARDIZE,
        ),
        (
            ScaleKernel(base_kernel=RBFKernel(ard=False)),
            ScalerEnum.STANDARDIZE,
            ScalerEnum.STANDARDIZE,
        ),
        (
            ScaleKernel(base_kernel=RBFKernel(ard=False)),
            ScalerEnum.IDENTITY,
            ScalerEnum.IDENTITY,
        ),
    ],
)
def test_SingleTaskGPModel_mordred(kernel, scaler, output_scaler):
    inputs = Inputs(features=[MolecularInput(key="x_mol")])
    outputs = Outputs(features=[ContinuousOutput(key="y")])
    experiments = [
        ["CC(=O)Oc1ccccc1C(=O)O", 88.0],
        ["c1ccccc1", 35.0],
        ["[CH3][CH2][OH]", 69.0],
        ["N[C@](C)(F)C(=O)O", 20.0],
    ]
    experiments = pd.DataFrame(experiments, columns=["x_mol", "y"])
    experiments["valid_y"] = 1
    model = SingleTaskGPSurrogate(
        inputs=inputs,
        outputs=outputs,
        kernel=kernel,
        scaler=scaler,
        output_scaler=output_scaler,
        input_preprocessing_specs={
            "x_mol": MordredDescriptors(descriptors=["NssCH2", "ATSC2d"]),
        },
    )
    model = surrogates.map(model)
    model.fit(experiments)
    # dump the model
    dump = model.dumps()
    # make predictions
    preds = model.predict(experiments.iloc[:-1])
    assert preds.shape == (3, 2)
    # check that model is composed correctly
    assert isinstance(model.model, SingleTaskGP)
    if output_scaler == ScalerEnum.STANDARDIZE:
        assert isinstance(model.model.outcome_transform, Standardize)
    elif output_scaler == ScalerEnum.IDENTITY:
        assert not hasattr(model.model, "outcome_transform")
    if scaler == ScalerEnum.NORMALIZE:
        assert isinstance(model.model.input_transform, Normalize)
    elif scaler == ScalerEnum.STANDARDIZE:
        assert isinstance(model.model.input_transform, InputStandardize)
    else:
        with pytest.raises(
            AttributeError,
            match="'SingleTaskGP' object has no attribute 'input_transform'",
        ):
            assert model.model.input_transform is None
    assert model.is_compatibilized is False
    # reload the model from dump and check for equality in predictions
    model2 = SingleTaskGPSurrogate(
        inputs=inputs,
        outputs=outputs,
        kernel=kernel,
        scaler=scaler,
        output_scaler=output_scaler,
        input_preprocessing_specs={
            "x_mol": MordredDescriptors(descriptors=["NssCH2", "ATSC2d"]),
        },
    )
    model2 = surrogates.map(model2)
    model2.loads(dump)
    preds2 = model2.predict(experiments.iloc[:-1])
    assert_frame_equal(preds, preds2)


@pytest.mark.parametrize("target_metric", list(RegressionMetricsEnum))
def test_hyperconfig_domain(target_metric: RegressionMetricsEnum):
    # we test here also the abstract methods from the corresponding base class
    # should be move somewhere else when tidying up all tests
    hy = SingleTaskGPHyperconfig(target_metric=target_metric)
    assert hy.domain.inputs == hy.inputs
    assert hy.domain.outputs.get_keys() == [target_metric.name]
    assert hy.domain.outputs[0].objective == metrics2objectives[target_metric]()


def test_hyperconfig_invalid():
    with pytest.raises(
        ValueError,
        match="It is not allowed to specify the number of its for FractionalFactorialStrategy",
    ):
        SingleTaskGPHyperconfig(n_iterations=5)
    with pytest.raises(
        ValueError,
        match="At least number of hyperparams plus 2 iterations has to be specified",
    ):
        SingleTaskGPHyperconfig(n_iterations=3, hyperstrategy="RandomStrategy")
    hy = SingleTaskGPHyperconfig(n_iterations=None, hyperstrategy="RandomStrategy")
    assert hy.n_iterations == 14


def test_SingleTaskGPHyperconfig():
    # we test here also the basic trainable
    benchmark = Himmelblau()
    surrogate_data_no_hy = SingleTaskGPSurrogate(
        inputs=benchmark.domain.inputs,
        outputs=benchmark.domain.outputs,
        hyperconfig=None,
    )
    with pytest.raises(ValueError, match="No hyperconfig available."):
        surrogate_data_no_hy.update_hyperparameters(
            benchmark.domain.inputs.sample(1).loc[0],
        )
    # test that correct stuff is written
    surrogate_data = SingleTaskGPSurrogate(
        inputs=benchmark.domain.inputs,
        outputs=benchmark.domain.outputs,
    )
    candidate = surrogate_data.hyperconfig.inputs.sample(1).loc[0]
    surrogate_data.update_hyperparameters(candidate)
    # if hasattr(surrogate_data.kernel, "base_kernel"):
    base_kernel = (
        surrogate_data.kernel.base_kernel
        if hasattr(surrogate_data.kernel, "base_kernel")
        else surrogate_data.kernel
    )
    if candidate.scalekernel == "True":
        assert hasattr(surrogate_data.kernel, "base_kernel")
    else:
        assert not hasattr(surrogate_data.kernel, "base_kernel")
    if candidate.kernel == "matern_1.5":
        assert isinstance(base_kernel, MaternKernel)
        assert base_kernel.nu == 1.5
    elif candidate.kernel == "matern_2.5":
        assert isinstance(base_kernel, MaternKernel)
        assert base_kernel.nu == 2.5
    else:
        assert isinstance(base_kernel, RBFKernel)
    if candidate.prior == "mbo":
        assert surrogate_data.noise_prior == MBO_NOISE_PRIOR()
        if candidate.scalekernel == "True":
            assert surrogate_data.kernel.outputscale_prior == MBO_OUTPUTSCALE_PRIOR()
        assert base_kernel.lengthscale_prior == MBO_LENGTHCALE_PRIOR()
    elif candidate.prior == "threesix":
        assert surrogate_data.noise_prior == THREESIX_NOISE_PRIOR()
        if candidate.scalekernel == "True":
            assert surrogate_data.kernel.outputscale_prior == THREESIX_SCALE_PRIOR()
        assert base_kernel.lengthscale_prior == THREESIX_LENGTHSCALE_PRIOR()
    else:
        assert surrogate_data.noise_prior == HVARFNER_NOISE_PRIOR()
        if candidate.scalekernel == "True":
            assert surrogate_data.kernel.outputscale_prior == THREESIX_SCALE_PRIOR()
        assert base_kernel.lengthscale_prior == HVARFNER_LENGTHSCALE_PRIOR()


def test_SingleTaskGPModel_feature_subsets():
    """make an additive kernel using feature subsets for each kernel in the sum"""
    benchmark = Hartmann()
    bench_x = benchmark.domain.inputs.sample(12)
    bench_expts = pd.concat([bench_x, benchmark.f(bench_x)], axis=1)

    input_names = benchmark.domain.inputs.get_keys()
    inputs_kernel_1 = input_names[:2]
    inputs_kernel_2 = input_names[2:]

    gp_data = SingleTaskGPSurrogate(
        inputs=benchmark.domain.inputs,
        outputs=benchmark.domain.outputs,
        kernel=AdditiveKernel(
            kernels=[
                RBFKernel(
                    ard=True,
                    lengthscale_prior=HVARFNER_LENGTHSCALE_PRIOR(),
                    features=inputs_kernel_1,
                ),
                RBFKernel(
                    ard=True,
                    lengthscale_prior=HVARFNER_LENGTHSCALE_PRIOR(),
                    features=inputs_kernel_2,
                ),
            ]
        ),
    )

    gp_mapped = surrogates.map(gp_data)
    assert hasattr(gp_mapped, "fit")
    assert len(gp_mapped.kernel.kernels) == 2
    assert gp_mapped.kernel.kernels[0].features == ["x_0", "x_1"]
    assert gp_mapped.kernel.kernels[1].features == ["x_2", "x_3", "x_4", "x_5"]
    gp_mapped.fit(bench_expts)
    pred = gp_mapped.predict(bench_expts)
    assert pred.shape == (12, 2)
    assert gp_mapped.model.covar_module.kernels[0].active_dims.tolist() == [0, 1]
    assert gp_mapped.model.covar_module.kernels[1].active_dims.tolist() == [2, 3, 4, 5]


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
def test_SingleTaskGPModel_mixed_features():
    """test that we can use a single task gp with mixed features"""
    inputs = Inputs(
        features=[
            ContinuousInput(key="x_1", bounds=(-4, 4)),
            ContinuousInput(key="x_2", bounds=(-4, 4)),
            CategoricalInput(key="x_cat_1", categories=["mama", "papa"]),
            CategoricalInput(key="x_cat_2", categories=["cat", "dog"]),
            MolecularInput(key="x_mol"),
        ],
    )
    outputs = Outputs(features=[ContinuousOutput(key="y")])

    experiment_values = [
        [2.56, -1.42, "papa", "dog", -3.98, 1, "CC(=O)Oc1ccccc1C(=O)O"],
        [3.84, -2.73, "mama", "cat", -197.46, 1, "c1ccccc1"],
        [3.57, 3.23, "papa", "cat", -74.55, 1, "[CH3][CH2][OH]"],
        [-0.07, -1.55, "mama", "dog", -179.14, 1, "N[C@](C)(F)C(=O)O"],
    ]
    experiments = pd.DataFrame(
        experiment_values,
        columns=["x_1", "x_2", "x_cat_1", "x_cat_2", "y", "valid_y", "x_mol"],
    )

    gp_data = SingleTaskGPSurrogate(
        inputs=inputs,
        outputs=outputs,
        kernel=AdditiveKernel(
            kernels=[
                HammingDistanceKernel(
                    ard=True,
                    features=["x_cat_1", "x_cat_2"],
                ),
                RBFKernel(
                    ard=True,
                    lengthscale_prior=HVARFNER_LENGTHSCALE_PRIOR(),
                    features=["x_1", "x_2"],
                ),
                TanimotoKernel(features=["x_mol"]),
            ]
        ),
    )

    gp_mapped = surrogates.map(gp_data)
    gp_mapped.fit(experiments)
    pred = gp_mapped.predict(experiments)
    assert pred.shape == (4, 2)
    assert gp_mapped.model.covar_module.kernels[0].active_dims.tolist() == [
        2050,
        2051,
        2052,
        2053,
    ]
    assert gp_mapped.model.covar_module.kernels[1].active_dims.tolist() == [0, 1]
    assert gp_mapped.model.covar_module.kernels[2].active_dims.tolist() == list(
        range(2, 2050)
    )
    # assert (pred['y_pred'] - experiments['y']).abs().mean() < 0.4


def test_MixedSingleTaskGPHyperconfig():
    inputs = Inputs(
        features=[
            ContinuousInput(
                key=f"x_{i + 1}",
                bounds=(-4, 4),
            )
            for i in range(2)
        ]
        + [CategoricalInput(key="x_cat", categories=["mama", "papa"])],
    )
    outputs = Outputs(features=[ContinuousOutput(key="y")])
    surrogate_data = MixedSingleTaskGPSurrogate(
        inputs=inputs,
        outputs=outputs,
    )
    candidate = surrogate_data.hyperconfig.inputs.sample(1).loc[0]
    surrogate_data.update_hyperparameters(candidate)
    assert surrogate_data.continuous_kernel.ard == (candidate["ard"] == "True")
    if candidate.continuous_kernel == "matern_1.5":
        assert isinstance(surrogate_data.continuous_kernel, MaternKernel)
        assert surrogate_data.continuous_kernel.nu == 1.5
    elif candidate.continuous_kernel == "matern_2.5":
        assert isinstance(surrogate_data.continuous_kernel, MaternKernel)
        assert surrogate_data.continuous_kernel.nu == 2.5
    else:
        assert isinstance(surrogate_data.continuous_kernel, RBFKernel)
    if candidate.prior == "mbo":
        assert surrogate_data.noise_prior == MBO_NOISE_PRIOR()
        assert (
            surrogate_data.continuous_kernel.lengthscale_prior == MBO_LENGTHCALE_PRIOR()
        )
    if candidate.prior == "threesix":
        assert surrogate_data.noise_prior == THREESIX_NOISE_PRIOR()
        assert (
            surrogate_data.continuous_kernel.lengthscale_prior
            == THREESIX_LENGTHSCALE_PRIOR()
        )
    if candidate.prior == "hvarfner":
        assert surrogate_data.noise_prior == HVARFNER_NOISE_PRIOR()
        assert (
            surrogate_data.continuous_kernel.lengthscale_prior
            == HVARFNER_LENGTHSCALE_PRIOR()
        )


def test_MixedSingleTaskGPModel_invalid_preprocessing():
    inputs = Inputs(
        features=[
            ContinuousInput(
                key=f"x_{i + 1}",
                bounds=(-4, 4),
            )
            for i in range(2)
        ],
    )
    outputs = Outputs(features=[ContinuousOutput(key="y")])
    experiments = inputs.sample(n=10)
    experiments.eval("y=((x_1**2 + x_2 - 11)**2+(x_1 + x_2**2 -7)**2)", inplace=True)
    experiments["valid_y"] = 1
    with pytest.raises(ValidationError):
        MixedSingleTaskGPSurrogate(
            inputs=inputs,
            outputs=outputs,
        )


@pytest.mark.parametrize(
    "kernel, scaler, output_scaler",
    [
        (RBFKernel(ard=True), ScalerEnum.NORMALIZE, ScalerEnum.STANDARDIZE),
        (RBFKernel(ard=False), ScalerEnum.STANDARDIZE, ScalerEnum.STANDARDIZE),
        (RBFKernel(ard=False), ScalerEnum.IDENTITY, ScalerEnum.IDENTITY),
    ],
)
def test_MixedSingleTaskGPModel(kernel, scaler, output_scaler):
    inputs = Inputs(
        features=[
            ContinuousInput(
                key=f"x_{i + 1}",
                bounds=(-4, 4),
            )
            for i in range(2)
        ]
        + [CategoricalInput(key="x_cat", categories=["mama", "papa"])],
    )
    outputs = Outputs(features=[ContinuousOutput(key="y")])
    experiments = inputs.sample(n=10)
    experiments.eval("y=((x_1**2 + x_2 - 11)**2+(x_1 + x_2**2 -7)**2)", inplace=True)
    experiments.loc[experiments.x_cat == "mama", "y"] *= 5.0
    experiments.loc[experiments.x_cat == "papa", "y"] /= 2.0
    experiments["valid_y"] = 1

    model = MixedSingleTaskGPSurrogate(
        inputs=inputs,
        outputs=outputs,
        input_preprocessing_specs={"x_cat": CategoricalEncodingEnum.ONE_HOT},
        scaler=scaler,
        output_scaler=output_scaler,
        continuous_kernel=kernel,
        categorical_kernel=HammingDistanceKernel(),
    )
    model = surrogates.map(model)
    with pytest.raises(ValueError):
        model.dumps()
    model.fit(experiments)
    # dump the model
    dump = model.dumps()
    # make predictions
    samples = inputs.sample(5)
    preds = model.predict(samples)
    assert preds.shape == (5, 2)
    # check that model is composed correctly
    assert isinstance(model.model, MixedSingleTaskGP)
    if output_scaler == ScalerEnum.STANDARDIZE:
        assert isinstance(model.model.outcome_transform, Standardize)
    elif output_scaler == ScalerEnum.IDENTITY:
        assert not hasattr(model.model, "outcome_transform")
    if scaler == ScalerEnum.NORMALIZE:
        assert isinstance(model.model.input_transform, ChainedInputTransform)
        assert isinstance(model.model.input_transform.tf1, Normalize)
        assert torch.eq(
            model.model.input_transform.tf1.indices,
            torch.tensor([0, 1], dtype=torch.int64),
        ).all()
        assert isinstance(model.model.input_transform.tf2, OneHotToNumeric)
    elif scaler == ScalerEnum.STANDARDIZE:
        assert isinstance(model.model.input_transform, ChainedInputTransform)
        assert isinstance(model.model.input_transform.tf1, InputStandardize)
        assert torch.eq(
            model.model.input_transform.tf1.indices,
            torch.tensor([0, 1], dtype=torch.int64),
        ).all()
        assert isinstance(model.model.input_transform.tf2, OneHotToNumeric)
    else:
        assert isinstance(model.model.input_transform, OneHotToNumeric)
    assert model.is_compatibilized is False
    # reload the model from dump and check for equality in predictions
    model2 = MixedSingleTaskGPSurrogate(
        inputs=inputs,
        outputs=outputs,
        continuous_kernel=kernel,
        scaler=scaler,
        output_scaler=output_scaler,
    )
    model2 = surrogates.map(model2)
    model2.loads(dump)
    preds2 = model2.predict(samples)
    assert_frame_equal(preds, preds2)


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
@pytest.mark.parametrize(
    "kernel, scaler, output_scaler",
    [
        (RBFKernel(ard=True), ScalerEnum.NORMALIZE, ScalerEnum.STANDARDIZE),
        (RBFKernel(ard=False), ScalerEnum.STANDARDIZE, ScalerEnum.STANDARDIZE),
        (RBFKernel(ard=False), ScalerEnum.IDENTITY, ScalerEnum.IDENTITY),
    ],
)
def test_MixedSingleTaskGPModel_mordred(kernel, scaler, output_scaler):
    inputs = Inputs(
        features=[MolecularInput(key="x_mol")]
        + [CategoricalInput(key="x_cat", categories=["a", "b"])],
    )
    outputs = Outputs(features=[ContinuousOutput(key="y")])
    experiments = [
        ["CC(=O)Oc1ccccc1C(=O)O", "a", 88.0],
        ["c1ccccc1", "a", 35.0],
        ["[CH3][CH2][OH]", "b", 69.0],
        ["N[C@](C)(F)C(=O)O", "b", 20.0],
    ]
    experiments = pd.DataFrame(experiments, columns=["x_mol", "x_cat", "y"])
    experiments["valid_y"] = 1
    model = MixedSingleTaskGPSurrogate(
        inputs=inputs,
        outputs=outputs,
        scaler=scaler,
        output_scaler=output_scaler,
        continuous_kernel=kernel,
        categorical_kernel=HammingDistanceKernel(),
        input_preprocessing_specs={
            "x_mol": MordredDescriptors(descriptors=["NssCH2", "ATSC2d"]),
            "x_cat": CategoricalEncodingEnum.ONE_HOT,
        },
    )
    model = surrogates.map(model)
    model.fit(experiments)
    # dump the model
    dump = model.dumps()
    # make predictions
    preds = model.predict(experiments.iloc[:-1])
    assert preds.shape == (3, 2)
    # check that model is composed correctly
    assert isinstance(model.model, MixedSingleTaskGP)
    if output_scaler == ScalerEnum.STANDARDIZE:
        assert isinstance(model.model.outcome_transform, Standardize)
    elif output_scaler == ScalerEnum.IDENTITY:
        assert not hasattr(model.model, "outcome_transform")
    if scaler == ScalerEnum.NORMALIZE:
        assert isinstance(model.model.input_transform, ChainedInputTransform)
        assert isinstance(model.model.input_transform.tf1, Normalize)
        assert torch.eq(
            model.model.input_transform.tf1.indices,
            torch.tensor([0, 1], dtype=torch.int64),
        ).all()
        assert isinstance(model.model.input_transform.tf2, OneHotToNumeric)
    elif scaler == ScalerEnum.STANDARDIZE:
        assert isinstance(model.model.input_transform, ChainedInputTransform)
        assert isinstance(model.model.input_transform.tf1, InputStandardize)
        assert torch.eq(
            model.model.input_transform.tf1.indices,
            torch.tensor([0, 1], dtype=torch.int64),
        ).all()
        assert isinstance(model.model.input_transform.tf2, OneHotToNumeric)
    else:
        assert isinstance(model.model.input_transform, OneHotToNumeric)
    assert model.is_compatibilized is False
    # reload the model from dump and check for equality in predictions
    model2 = MixedSingleTaskGPSurrogate(
        inputs=inputs,
        outputs=outputs,
        continuous_kernel=kernel,
        scaler=scaler,
        output_scaler=output_scaler,
        input_preprocessing_specs={
            "x_mol": MordredDescriptors(descriptors=["NssCH2", "ATSC2d"]),
            "x_cat": CategoricalEncodingEnum.ONE_HOT,
        },
    )
    model2 = surrogates.map(model2)
    model2.loads(dump)
    preds2 = model2.predict(experiments.iloc[:-1])
    assert_frame_equal(preds, preds2)


@pytest.mark.parametrize(
    "kernel, scaler, output_scaler",
    [
        (
            ScaleKernel(base_kernel=RBFKernel(ard=True)),
            ScalerEnum.NORMALIZE,
            ScalerEnum.STANDARDIZE,
        ),
        (
            ScaleKernel(base_kernel=RBFKernel(ard=False)),
            ScalerEnum.STANDARDIZE,
            ScalerEnum.STANDARDIZE,
        ),
        (
            ScaleKernel(base_kernel=RBFKernel(ard=False)),
            ScalerEnum.IDENTITY,
            ScalerEnum.IDENTITY,
        ),
    ],
)
def test_RobustSingleTaskGPModel(kernel, scaler, output_scaler):
    inputs = Inputs(
        features=[
            ContinuousInput(
                key=f"x_{i + 1}",
                bounds=(-4, 4),
            )
            for i in range(2)
        ],
    )
    outputs = Outputs(features=[ContinuousOutput(key="y")])
    experiments = inputs.sample(n=10)
    experiments.eval("y=((x_1**2 + x_2 - 11)**2+(x_1 + x_2**2 -7)**2)", inplace=True)
    experiments["valid_y"] = 1
    model = RobustSingleTaskGPSurrogate(
        inputs=inputs,
        outputs=outputs,
        kernel=kernel,
        scaler=scaler,
        output_scaler=output_scaler,
    )
    model = surrogates.map(model)
    samples = inputs.sample(5)
    # test error on non fitted model
    with pytest.raises(ValueError):
        model.predict(samples)
    model.fit(experiments)
    # dump the model
    dump = model.dumps()
    # make predictions
    samples2 = samples.copy()
    samples2 = samples2.astype({"x_1": "object"})
    preds = model.predict(samples2)
    assert preds.shape == (5, 2)
    # check that model is composed correctly
    assert isinstance(model.model, RobustRelevancePursuitSingleTaskGP)
    if output_scaler == ScalerEnum.STANDARDIZE:
        assert isinstance(model.model.outcome_transform, Standardize)
    elif output_scaler == ScalerEnum.IDENTITY:
        assert not hasattr(model.model, "outcome_transform")
    if scaler == ScalerEnum.NORMALIZE:
        assert isinstance(model.model.input_transform, Normalize)
    elif scaler == ScalerEnum.STANDARDIZE:
        assert isinstance(model.model.input_transform, InputStandardize)
    else:
        with pytest.raises(AttributeError):
            assert model.model.input_transform is None
    assert model.is_compatibilized is False
    # reload the model from dump and check for equality in predictions
    model2 = RobustSingleTaskGPSurrogate(
        inputs=inputs,
        outputs=outputs,
        kernel=kernel,
        scaler=scaler,
    )
    model2 = surrogates.map(model2)
    model2.loads(dump)
    preds2 = model2.predict(samples)

    assert_frame_equal(preds, preds2)

    model3 = RobustSingleTaskGPSurrogate(
        inputs=inputs,
        outputs=outputs,
        kernel=kernel,
        scaler=scaler,
    )

    model3 = surrogates.map(model3)
    # test predict outliers
    preds_outliers = model3.predict_outliers(experiments)

    # assert that preds_outliers dataframe had the same length as experiments
    assert len(preds_outliers) == len(experiments)

    # check for the correct columns
    assert set(preds_outliers.columns) == {"y_pred", "y_sd", "y_rho"}


def test_RobustSingleTaskGPHyperconfig():
    # we test here also the basic trainable
    benchmark = Himmelblau()
    surrogate_data_no_hy = RobustSingleTaskGPSurrogate(
        inputs=benchmark.domain.inputs,
        outputs=benchmark.domain.outputs,
        hyperconfig=None,
    )
    with pytest.raises(ValueError, match="No hyperconfig available."):
        surrogate_data_no_hy.update_hyperparameters(
            benchmark.domain.inputs.sample(1).loc[0],
        )
    # test that correct stuff is written
    surrogate_data = RobustSingleTaskGPSurrogate(
        inputs=benchmark.domain.inputs,
        outputs=benchmark.domain.outputs,
    )

    assert (
        surrogate_data.kernel.lengthscale_constraint
        == ROBUSTGP_LENGTHSCALE_CONSTRAINT()
    )

    candidate = surrogate_data.hyperconfig.inputs.sample(1).loc[0]
    # surrogate_data.update_hyperparameters(candidate, lengthscale_constraint=ROBUSTGP_LENGTHSCALE_CONSTRAINT(), outputscale_constraint=ROBUSTGP_OUTPUTSCALE_CONSTRAINT())
    surrogate_data.update_hyperparameters(candidate)
    if hasattr(surrogate_data.kernel, "base_kernel"):
        # if surrogate_data.kernel == ScaleKernel():
        #     assert surrogate_data.kernel.outputscale_constraint == ROBUSTGP_OUTPUTSCALE_CONSTRAINT()
        assert surrogate_data.kernel.base_kernel.ard == (candidate["ard"] == "True")
        # assert surrogate_data.kernel.base_kernel.lengthscale_constraint == ROBUSTGP_LENGTHSCALE_CONSTRAINT()
        if candidate.kernel == "matern_1.5":
            assert isinstance(surrogate_data.kernel.base_kernel, MaternKernel)
            assert surrogate_data.kernel.base_kernel.nu == 1.5
        elif candidate.kernel == "matern_2.5":
            assert isinstance(surrogate_data.kernel.base_kernel, MaternKernel)
            assert surrogate_data.kernel.base_kernel.nu == 2.5
        else:
            assert isinstance(surrogate_data.kernel.base_kernel, RBFKernel)
        if candidate.prior == "mbo":
            assert surrogate_data.noise_prior == MBO_NOISE_PRIOR()
            assert surrogate_data.kernel.outputscale_prior == MBO_OUTPUTSCALE_PRIOR()
            assert (
                surrogate_data.kernel.base_kernel.lengthscale_prior
                == MBO_LENGTHCALE_PRIOR()
            )
        elif candidate.prior == "threesix":
            assert surrogate_data.noise_prior == THREESIX_NOISE_PRIOR()
            assert surrogate_data.kernel.outputscale_prior == THREESIX_SCALE_PRIOR()
            assert (
                surrogate_data.kernel.base_kernel.lengthscale_prior
                == THREESIX_LENGTHSCALE_PRIOR()
            )
        else:
            assert surrogate_data.noise_prior == HVARFNER_NOISE_PRIOR()
            assert surrogate_data.kernel.outputscale_prior == THREESIX_SCALE_PRIOR()
            assert (
                surrogate_data.kernel.base_kernel.lengthscale_prior
                == HVARFNER_LENGTHSCALE_PRIOR()
            )
