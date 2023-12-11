import importlib

import pandas as pd
import pytest
import torch
from botorch.models import MixedSingleTaskGP, SingleTaskGP
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
from bofire.benchmarks.api import Himmelblau
from bofire.data_models.domain.api import Inputs, Outputs
from bofire.data_models.enum import CategoricalEncodingEnum, RegressionMetricsEnum
from bofire.data_models.features.api import (
    CategoricalDescriptorInput,
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
    MolecularInput,
)
from bofire.data_models.kernels.api import (
    HammondDistanceKernel,
    MaternKernel,
    RBFKernel,
    ScaleKernel,
    TanimotoKernel,
)
from bofire.data_models.molfeatures.api import (
    Fingerprints,
    FingerprintsFragments,
    Fragments,
    MordredDescriptors,
)
from bofire.data_models.priors.api import (
    BOTORCH_LENGTHCALE_PRIOR,
    BOTORCH_NOISE_PRIOR,
    BOTORCH_SCALE_PRIOR,
    MBO_LENGTHCALE_PRIOR,
    MBO_NOISE_PRIOR,
    MBO_OUTPUTSCALE_PRIOR,
)
from bofire.data_models.surrogates.api import (
    MixedSingleTaskGPSurrogate,
    ScalerEnum,
    SingleTaskGPHyperconfig,
    SingleTaskGPSurrogate,
)
from bofire.data_models.surrogates.mixed_tanimoto_gp import MixedTanimotoGPSurrogate
from bofire.data_models.surrogates.tanimoto_gp import TanimotoGPSurrogate
from bofire.data_models.surrogates.trainable import metrics2objectives
from bofire.surrogates.mixed_tanimoto_gp import MixedTanimotoGP
from bofire.surrogates.single_task_gp import get_scaler
from bofire.utils.torch_tools import tkwargs

RDKIT_AVAILABLE = importlib.util.find_spec("rdkit") is not None


@pytest.mark.parametrize(
    "scaler_enum, input_preprocessing_specs, expected_scaler, expected_indices, expected_offset, expected_coefficient",
    [
        (
            ScalerEnum.NORMALIZE,
            {
                "x_cat": CategoricalEncodingEnum.ONE_HOT,
                "x_desc": CategoricalEncodingEnum.ONE_HOT,
            },
            Normalize,
            torch.tensor([0, 1], dtype=torch.int64),
            torch.tensor([-4.0, -4.0]).to(**tkwargs),
            torch.tensor([8.0, 8.0]).to(**tkwargs),
        ),
        (
            ScalerEnum.NORMALIZE,
            {
                "x_cat": CategoricalEncodingEnum.ONE_HOT,
                "x_desc": CategoricalEncodingEnum.DESCRIPTOR,
            },
            Normalize,
            torch.tensor([0, 1, 2], dtype=torch.int64),
            torch.tensor([-4.0, -4.0, 1.0]).to(**tkwargs),
            torch.tensor([8.0, 8.0, 5.0]).to(**tkwargs),
        ),
        (
            ScalerEnum.STANDARDIZE,
            {
                "x_cat": CategoricalEncodingEnum.ONE_HOT,
                "x_desc": CategoricalEncodingEnum.ONE_HOT,
            },
            InputStandardize,
            torch.tensor([0, 1], dtype=torch.int64),
            None,
            None,
        ),
        (
            ScalerEnum.STANDARDIZE,
            {
                "x_cat": CategoricalEncodingEnum.ONE_HOT,
                "x_desc": CategoricalEncodingEnum.DESCRIPTOR,
            },
            InputStandardize,
            torch.tensor([0, 1, 2], dtype=torch.int64),
            None,
            None,
        ),
        (
            ScalerEnum.IDENTITY,
            {
                "x_cat": CategoricalEncodingEnum.ONE_HOT,
                "x_desc": CategoricalEncodingEnum.ONE_HOT,
            },
            type(None),
            None,
            None,
            None,
        ),
        (
            ScalerEnum.IDENTITY,
            {
                "x_cat": CategoricalEncodingEnum.ONE_HOT,
                "x_desc": CategoricalEncodingEnum.DESCRIPTOR,
            },
            type(None),
            None,
            None,
            None,
        ),
    ],
)
def test_get_scaler(
    scaler_enum,
    input_preprocessing_specs,
    expected_scaler,
    expected_indices,
    expected_offset,
    expected_coefficient,
):
    inputs = Inputs(
        features=[
            ContinuousInput(
                key=f"x_{i+1}",
                bounds=(-4, 4),
            )
            for i in range(2)
        ]
        + [
            CategoricalInput(key="x_cat", categories=["mama", "papa"]),
            CategoricalDescriptorInput(
                key="x_desc",
                categories=["alpha", "beta"],
                descriptors=["oskar"],
                values=[[1], [6]],
            ),
        ]
    )
    experiments = inputs.sample(n=10)
    scaler = get_scaler(
        inputs=inputs,
        input_preprocessing_specs=input_preprocessing_specs,
        scaler=scaler_enum,
        X=experiments[inputs.get_keys()],
    )
    assert isinstance(scaler, expected_scaler)
    if expected_indices is not None:
        assert (scaler.indices == expected_indices).all()
    else:
        with pytest.raises(AttributeError):
            assert (scaler.indices == expected_indices).all()
    if expected_offset is not None:
        assert torch.allclose(scaler.offset, expected_offset)
        assert torch.allclose(scaler.coefficient, expected_coefficient)
    else:
        if scaler is None:
            with pytest.raises(AttributeError):
                assert (scaler.offset == expected_offset).all()
            with pytest.raises(AttributeError):
                assert (scaler.coefficient == expected_coefficient).all()


@pytest.mark.parametrize(
    "scaler_enum, input_preprocessing_specs, expected_scaler, expected_indices_length",
    [
        (
            ScalerEnum.NORMALIZE,
            {
                "x_mol": MordredDescriptors(descriptors=["NssCH2", "ATSC2d"]),
            },
            Normalize,
            4,
        ),
        (
            ScalerEnum.NORMALIZE,
            {
                "x_mol": Fingerprints(n_bits=32),
            },
            Normalize,
            2,
        ),
        (
            ScalerEnum.STANDARDIZE,
            {
                "x_mol": MordredDescriptors(descriptors=["NssCH2", "ATSC2d"]),
            },
            InputStandardize,
            4,
        ),
        (
            ScalerEnum.STANDARDIZE,
            {
                "x_mol": Fragments(),
            },
            InputStandardize,
            2,
        ),
        (
            ScalerEnum.IDENTITY,
            {
                "x_mol": MordredDescriptors(descriptors=["NssCH2", "ATSC2d"]),
            },
            type(None),
            0,
        ),
        (
            ScalerEnum.IDENTITY,
            {
                "x_mol": FingerprintsFragments(n_bits=32),
            },
            type(None),
            0,
        ),
    ],
)
def test_get_scaler_molecular(
    scaler_enum,
    input_preprocessing_specs,
    expected_scaler,
    expected_indices_length,
):
    inputs = Inputs(
        features=[
            ContinuousInput(
                key=f"x_{i+1}",
                bounds=(0, 5),
            )
            for i in range(2)
        ]
        + [MolecularInput(key="x_mol")]
    )
    experiments = [
        [5.0, 2.5, "CC(=O)Oc1ccccc1C(=O)O"],
        [4.0, 2.0, "c1ccccc1"],
        [3.0, 0.5, "[CH3][CH2][OH]"],
        [1.5, 4.5, "N[C@](C)(F)C(=O)O"],
    ]
    experiments = pd.DataFrame(experiments, columns=["x_1", "x_2", "x_mol"])
    scaler = get_scaler(
        inputs=inputs,
        input_preprocessing_specs=input_preprocessing_specs,
        scaler=scaler_enum,
        X=experiments[inputs.get_keys()],
    )
    assert isinstance(scaler, expected_scaler)
    if expected_indices_length != 0:
        assert len(scaler.indices) == expected_indices_length
    else:
        with pytest.raises(
            AttributeError, match="'NoneType' object has no attribute 'indices'"
        ):
            assert scaler.indices is None


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
                key=f"x_{i+1}",
                bounds=(-4, 4),
            )
            for i in range(2)
        ]
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
    inputs = Inputs(
        features=[MolecularInput(key="x_mol")]
    )
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
                "x_mol": MordredDescriptors(descriptors=["NssCH2", "ATSC2d"])
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
                "x_mol": MordredDescriptors(descriptors=["NssCH2", "ATSC2d"])
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
        match="It is not allowed to scpecify the number of its for FactorialStrategy",
    ):
        SingleTaskGPHyperconfig(n_iterations=5)
    with pytest.raises(
        ValueError,
        match="At least number of hyperparams plus 2 iterations has to be specified",
    ):
        SingleTaskGPHyperconfig(n_iterations=3, hyperstrategy="RandomStrategy")
    hy = SingleTaskGPHyperconfig(n_iterations=None, hyperstrategy="RandomStrategy")
    assert hy.n_iterations == 13


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
            benchmark.domain.inputs.sample(1).loc[0]
        )
    # test that correct stuff is written
    surrogate_data = SingleTaskGPSurrogate(
        inputs=benchmark.domain.inputs, outputs=benchmark.domain.outputs
    )
    candidate = surrogate_data.hyperconfig.inputs.sample(1).loc[0]
    surrogate_data.update_hyperparameters(candidate)
    assert surrogate_data.kernel.base_kernel.ard == (candidate["ard"] == "True")
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
    else:
        assert surrogate_data.noise_prior == BOTORCH_NOISE_PRIOR()
        assert surrogate_data.kernel.outputscale_prior == BOTORCH_SCALE_PRIOR()
        assert (
            surrogate_data.kernel.base_kernel.lengthscale_prior
            == BOTORCH_LENGTHCALE_PRIOR()
        )


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
@pytest.mark.parametrize(
    "kernel, specs",
    [
        (
            ScaleKernel(base_kernel=TanimotoKernel(ard=False)),
            {"x_1": Fingerprints(n_bits=32)},
        ),
        (
            ScaleKernel(base_kernel=TanimotoKernel(ard=True)),
            {"x_1": Fragments()},
        ),
        (
            ScaleKernel(base_kernel=TanimotoKernel(ard=False)),
            {"x_1": FingerprintsFragments(n_bits=32)},
        ),
    ],
)
def test_TanimotoGP(kernel, specs):
    inputs = Inputs(features=[MolecularInput(key="x_1")])
    outputs = Outputs(features=[ContinuousOutput(key="y")])
    experiments = [
        ["CC(=O)Oc1ccccc1C(=O)O", 88.0],
        ["c1ccccc1", 35.0],
        ["[CH3][CH2][OH]", 69.0],
        ["N[C@](C)(F)C(=O)O", 20.0],
    ]
    experiments = pd.DataFrame(experiments, columns=["x_1", "y"])
    experiments["valid_y"] = 1
    model = TanimotoGPSurrogate(
        inputs=inputs,
        outputs=outputs,
        kernel=kernel,
        input_preprocessing_specs=specs,
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
    assert isinstance(model.model.outcome_transform, Standardize)
    assert model.is_compatibilized is False
    # reload the model from dump and check for equality in predictions
    model2 = TanimotoGPSurrogate(
        inputs=inputs,
        outputs=outputs,
        kernel=kernel,
        input_preprocessing_specs=specs,
    )
    model2 = surrogates.map(model2)
    model2.loads(dump)
    preds2 = model2.predict(experiments.iloc[:-1])
    assert_frame_equal(preds, preds2)


def test_TanimotoGPModel_invalid_preprocessing_mordred():
    inputs = Inputs(features=[MolecularInput(key="x_mol")])
    outputs = Outputs(features=[ContinuousOutput(key="y")])
    experiments = [
        ["CC(=O)Oc1ccccc1C(=O)O", 1.5],
        ["c1ccccc1", 3.5],
        ["[CH3][CH2][OH]", 2.0],
        ["N[C@](C)(F)C(=O)O", 4.0],
    ]
    experiments = pd.DataFrame(experiments, columns=["x_mol", "y"])
    experiments["valid_y"] = 1
    with pytest.raises(
        ValidationError,
        match="TanimotoGPSurrogate can only be used if at least one of fingerprints, fragments, or fingerprintsfragments features are present.",
    ):
        TanimotoGPSurrogate(
            inputs=inputs,
            outputs=outputs,
            input_preprocessing_specs={
                "x_mol": MordredDescriptors(descriptors=["NssCH2", "ATSC2d"])
            },
        )


def test_MixedSingleTaskGPModel_invalid_preprocessing():
    inputs = Inputs(
        features=[
            ContinuousInput(
                key=f"x_{i+1}",
                bounds=(-4, 4),
            )
            for i in range(2)
        ]
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
                key=f"x_{i+1}",
                bounds=(-4, 4),
            )
            for i in range(2)
        ]
        + [CategoricalInput(key="x_cat", categories=["mama", "papa"])]
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
        categorical_kernel=HammondDistanceKernel(),
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
        features=[MolecularInput(key="x_mol")] +
        [CategoricalInput(key="x_cat", categories=["a", "b"])]
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
        categorical_kernel=HammondDistanceKernel(),
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


def test_MixedTanimotoGPModel_invalid_preprocessing():
    inputs = Inputs(
        features=[
            ContinuousInput(
                key=f"x_{i+1}",
                bounds=(-4, 4),
            )
            for i in range(2)
        ]
        + [CategoricalInput(key="x_cat", categories=["mama", "papa"])]
    )
    outputs = Outputs(features=[ContinuousOutput(key="y")])
    experiments = inputs.sample(n=10)
    experiments.eval("y=((x_1**2 + x_2 - 11)**2+(x_1 + x_2**2 -7)**2)", inplace=True)
    experiments.loc[experiments.x_cat == "mama", "y"] *= 5.0
    experiments.loc[experiments.x_cat == "papa", "y"] /= 2.0
    experiments["valid_y"] = 1
    with pytest.raises(
        ValidationError,
        match="MixedTanimotoGPSurrogate can only be used if at least one of fingerprints, fragments, or fingerprintsfragments features are present.",
    ):
        MixedTanimotoGPSurrogate(
            inputs=inputs,
            outputs=outputs,
        )


def test_MixedTanimotoGPModel_invalid_preprocessing_mordred():
    inputs = Inputs(
        features=[
            ContinuousInput(
                key=f"x_{i+1}",
                bounds=(0, 5),
            )
            for i in range(2)
        ]
        + [MolecularInput(key="x_mol")]
    )
    outputs = Outputs(features=[ContinuousOutput(key="y")])
    experiments = [
        [5.0, 2.5, "CC(=O)Oc1ccccc1C(=O)O", 1.5],
        [4.0, 2.0, "c1ccccc1", 3.5],
        [3.0, 0.5, "[CH3][CH2][OH]", 2.0],
        [1.5, 4.5, "N[C@](C)(F)C(=O)O", 4.0],
    ]
    experiments = pd.DataFrame(experiments, columns=["x_1", "x_2", "x_mol", "y"])
    experiments["valid_y"] = 1
    with pytest.raises(
        ValidationError,
        match="MixedTanimotoGPSurrogate can only be used if at least one of fingerprints, fragments, or fingerprintsfragments features are present.",
    ):
        MixedTanimotoGPSurrogate(
            inputs=inputs,
            outputs=outputs,
            input_preprocessing_specs={
                "x_mol": MordredDescriptors(descriptors=["NssCH2", "ATSC2d"])
            },
        )


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
@pytest.mark.parametrize(
    "kernel, specs, scaler",
    [
        (
            TanimotoKernel(ard=True),
            {"x_mol": Fingerprints(n_bits=32)},
            ScalerEnum.NORMALIZE,
        ),
        (
            TanimotoKernel(ard=False),
            {"x_mol": Fragments()},
            ScalerEnum.IDENTITY,
        ),
        (
            TanimotoKernel(ard=True),
            {"x_mol": FingerprintsFragments(n_bits=32)},
            ScalerEnum.STANDARDIZE,
        ),
    ],
)
def test_MixedTanimotoGP_continuous(kernel, specs, scaler):
    inputs = Inputs(
        features=[MolecularInput(key="x_mol")]
        + [
            ContinuousInput(
                key=f"x_{i+1}",
                bounds=(0, 5.0),
            )
            for i in range(2)
        ]
    )
    outputs = Outputs(features=[ContinuousOutput(key="y")])
    experiments = [
        ["CC(=O)Oc1ccccc1C(=O)O", 5.0, 2.5, 88.0],
        ["c1ccccc1", 4.0, 2.0, 35.0],
        ["[CH3][CH2][OH]", 3.0, 0.5, 69.0],
        ["N[C@](C)(F)C(=O)O", 1.5, 4.5, 20.0],
    ]
    experiments = pd.DataFrame(experiments, columns=["x_mol", "x_1", "x_2", "y"])
    experiments["valid_y"] = 1
    model = MixedTanimotoGPSurrogate(
        inputs=inputs,
        outputs=outputs,
        molecular_kernel=kernel,
        scaler=scaler,
        input_preprocessing_specs=specs,
    )
    model = surrogates.map(model)
    model.fit(experiments)
    # dump the model
    dump = model.dumps()
    # make predictions
    preds = model.predict(experiments.iloc[:-1])
    assert preds.shape == (3, 2)
    # check that model is composed correctly
    assert isinstance(model.model, MixedTanimotoGP)
    assert isinstance(model.model.outcome_transform, Standardize)
    if scaler == ScalerEnum.NORMALIZE:
        assert isinstance(model.model.input_transform, Normalize)
        assert model.model.input_transform.indices.shape == torch.Size([2])
    elif scaler == ScalerEnum.STANDARDIZE:
        assert isinstance(model.model.input_transform, InputStandardize)
        assert model.model.input_transform.indices.shape == torch.Size([2])
    else:
        with pytest.raises(
            AttributeError,
            match="'MixedTanimotoGP' object has no attribute 'input_transform'",
        ):
            assert model.model.input_transform is None
    assert model.is_compatibilized is False
    # reload the model from dump and check for equality in predictions
    model2 = MixedTanimotoGPSurrogate(
        inputs=inputs,
        outputs=outputs,
        molecular_kernel=kernel,
        input_preprocessing_specs=specs,
    )
    model2 = surrogates.map(model2)
    model2.loads(dump)
    preds2 = model2.predict(experiments.iloc[:-1])
    assert_frame_equal(preds, preds2)


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
@pytest.mark.parametrize(
    "kernel, specs, scaler",
    [
        (
            TanimotoKernel(ard=True),
            {
                "x_mol": Fingerprints(n_bits=32),
                "x_cat": CategoricalEncodingEnum.ONE_HOT,
            },
            ScalerEnum.NORMALIZE,
        ),
        (
            TanimotoKernel(ard=False),
            {"x_mol": Fragments(), "x_cat": CategoricalEncodingEnum.ONE_HOT},
            ScalerEnum.IDENTITY,
        ),
        (
            TanimotoKernel(ard=True),
            {
                "x_mol": FingerprintsFragments(n_bits=32),
                "x_cat": CategoricalEncodingEnum.ONE_HOT,
            },
            ScalerEnum.STANDARDIZE,
        ),
    ],
)
def test_MixedTanimotoGP(kernel, specs, scaler):
    inputs = Inputs(
        features=[MolecularInput(key="x_mol")]
        + [
            ContinuousInput(
                key="x_1",
                bounds=(0, 5.0),
            )
        ]
        + [CategoricalInput(key="x_cat", categories=["a", "b"])]
    )
    outputs = Outputs(features=[ContinuousOutput(key="y")])
    experiments = [
        ["CC(=O)Oc1ccccc1C(=O)O", 5.0, "a", 88.0],
        ["c1ccccc1", 4.0, "a", 35.0],
        ["[CH3][CH2][OH]", 3.0, "b", 69.0],
        ["N[C@](C)(F)C(=O)O", 1.5, "b", 20.0],
    ]
    experiments = pd.DataFrame(experiments, columns=["x_mol", "x_1", "x_cat", "y"])
    experiments["valid_y"] = 1
    model = MixedTanimotoGPSurrogate(
        inputs=inputs,
        outputs=outputs,
        molecular_kernel=kernel,
        scaler=scaler,
        input_preprocessing_specs=specs,
    )
    model = surrogates.map(model)
    model.fit(experiments)
    # dump the model
    dump = model.dumps()
    # make predictions
    preds = model.predict(experiments.iloc[:-1])
    assert preds.shape == (3, 2)
    # check that model is composed correctly
    assert isinstance(model.model, MixedTanimotoGP)
    assert isinstance(model.model.outcome_transform, Standardize)
    if scaler == ScalerEnum.NORMALIZE:
        assert isinstance(model.model.input_transform, ChainedInputTransform)
        assert isinstance(model.model.input_transform.tf1, Normalize)
        assert model.model.input_transform.tf1.indices.shape == torch.Size([1])
        assert isinstance(model.model.input_transform.tf2, OneHotToNumeric)
    elif scaler == ScalerEnum.STANDARDIZE:
        assert isinstance(model.model.input_transform, ChainedInputTransform)
        assert isinstance(model.model.input_transform.tf1, InputStandardize)
        assert model.model.input_transform.tf1.indices.shape == torch.Size([1])
        assert isinstance(model.model.input_transform.tf2, OneHotToNumeric)
    else:
        assert isinstance(model.model.input_transform, OneHotToNumeric)
    assert model.is_compatibilized is False
    # reload the model from dump and check for equality in predictions
    model2 = MixedTanimotoGPSurrogate(
        inputs=inputs,
        outputs=outputs,
        molecular_kernel=kernel,
        input_preprocessing_specs=specs,
    )
    model2 = surrogates.map(model2)
    model2.loads(dump)
    preds2 = model2.predict(experiments.iloc[:-1])
    assert_frame_equal(preds, preds2)


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
@pytest.mark.parametrize(
    "kernel, specs",
    [
        (
            TanimotoKernel(ard=True),
            {
                "x_mol": Fingerprints(n_bits=32),
                "x_cat": CategoricalEncodingEnum.ONE_HOT,
            },
        ),
        (
            TanimotoKernel(ard=False),
            {"x_mol": Fragments(), "x_cat": CategoricalEncodingEnum.ONE_HOT},
        ),
        (
            TanimotoKernel(ard=True),
            {
                "x_mol": FingerprintsFragments(n_bits=32),
                "x_cat": CategoricalEncodingEnum.ONE_HOT,
            },
        ),
    ],
)
def test_MixedTanimotoGP_categorical(kernel, specs):
    inputs = Inputs(
        features=[MolecularInput(key="x_mol")]
        + [CategoricalInput(key="x_cat", categories=["a", "b"])]
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
    model = MixedTanimotoGPSurrogate(
        inputs=inputs,
        outputs=outputs,
        molecular_kernel=kernel,
        input_preprocessing_specs=specs,
    )
    model = surrogates.map(model)
    model.fit(experiments)
    # dump the model
    dump = model.dumps()
    # make predictions
    preds = model.predict(experiments.iloc[:-1])
    assert preds.shape == (3, 2)
    # check that model is composed correctly
    assert isinstance(model.model, MixedTanimotoGP)
    assert isinstance(model.model.outcome_transform, Standardize)
    assert isinstance(model.model.input_transform, OneHotToNumeric)
    assert model.is_compatibilized is False
    # reload the model from dump and check for equality in predictions
    model2 = MixedTanimotoGPSurrogate(
        inputs=inputs,
        outputs=outputs,
        input_preprocessing_specs=specs,
    )
    model2 = surrogates.map(model2)
    model2.loads(dump)
    preds2 = model2.predict(experiments.iloc[:-1])
    assert_frame_equal(preds, preds2)


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
@pytest.mark.parametrize(
    "molecular_kernel, continuous_kernel, specs, scaler",
    [
        (
            TanimotoKernel(ard=True),
            RBFKernel(ard=True),
            {
                "x_1": Fingerprints(n_bits=32),
                "x_2": MordredDescriptors(descriptors=["NssCH2", "ATSC2d"]),
            },
            ScalerEnum.NORMALIZE,
        ),
        (
            TanimotoKernel(ard=False),
            MaternKernel(nu=0.5, ard=True),
            {
                "x_1": Fragments(),
                "x_2": MordredDescriptors(descriptors=["NssCH2", "ATSC2d"]),
            },
            ScalerEnum.IDENTITY,
        ),
        (
            TanimotoKernel(ard=True),
            MaternKernel(nu=2.5, ard=False),
            {
                "x_1": FingerprintsFragments(n_bits=32),
                "x_2": MordredDescriptors(descriptors=["NssCH2", "ATSC2d"]),
            },
            ScalerEnum.STANDARDIZE,
        ),
    ],
)
def test_MixedTanimotoGP_with_mordred(
    molecular_kernel, continuous_kernel, specs, scaler
):
    inputs = Inputs(
        features=[
            MolecularInput(
                key=f"x_{i+1}",
            )
            for i in range(2)
        ]
    )
    outputs = Outputs(features=[ContinuousOutput(key="y")])
    experiments = [
        ["CC(=O)Oc1ccccc1C(=O)O", "CC(=O)Oc1ccccc1C(=O)O", 88.0],
        ["c1ccccc1", "c1ccccc1", 35.0],
        ["[CH3][CH2][OH]", "[CH3][CH2][OH]", 69.0],
        ["N[C@](C)(F)C(=O)O", "N[C@](C)(F)C(=O)O", 20.0],
    ]
    experiments = pd.DataFrame(experiments, columns=["x_1", "x_2", "y"])
    experiments["valid_y"] = 1
    model = MixedTanimotoGPSurrogate(
        inputs=inputs,
        outputs=outputs,
        molecular_kernel=molecular_kernel,
        continuous_kernel=continuous_kernel,
        scaler=scaler,
        input_preprocessing_specs=specs,
    )
    model = surrogates.map(model)
    model.fit(experiments)
    # dump the model
    dump = model.dumps()
    # make predictions
    preds = model.predict(experiments.iloc[:-1])
    assert preds.shape == (3, 2)
    # check that model is composed correctly
    assert isinstance(model.model, MixedTanimotoGP)
    assert isinstance(model.model.outcome_transform, Standardize)
    if scaler == ScalerEnum.NORMALIZE:
        assert isinstance(model.model.input_transform, Normalize)
        assert model.model.input_transform.indices.shape == torch.Size([2])
    elif scaler == ScalerEnum.STANDARDIZE:
        assert isinstance(model.model.input_transform, InputStandardize)
        assert model.model.input_transform.indices.shape == torch.Size([2])
    else:
        with pytest.raises(
            AttributeError,
            match="'MixedTanimotoGP' object has no attribute 'input_transform'",
        ):
            assert model.model.input_transform is None
    assert model.is_compatibilized is False
    # reload the model from dump and check for equality in predictions
    model2 = MixedTanimotoGPSurrogate(
        inputs=inputs,
        outputs=outputs,
        molecular_kernel=molecular_kernel,
        continuous_kernel=continuous_kernel,
        input_preprocessing_specs=specs,
    )
    model2 = surrogates.map(model2)
    model2.loads(dump)
    preds2 = model2.predict(experiments.iloc[:-1])
    assert_frame_equal(preds, preds2)
