import numpy as np
import pandas as pd
import pytest

import bofire.data_models.outlier_detection.api as data_models
import bofire.outlier_detection.api as mapper
from bofire.data_models.domain.api import Inputs, Outputs
from bofire.data_models.enum import CategoricalEncodingEnum
from bofire.data_models.features.api import (
    CategoricalDescriptorInput,
    ContinuousInput,
    ContinuousOutput,
)
from bofire.data_models.kernels.api import RBFKernel, ScaleKernel
from bofire.data_models.surrogates.api import ScalerEnum, SingleTaskGPSurrogate
from bofire.outlier_detection.outlier_detections import OutlierDetections


def neal_func(x):
    return 0.3 + 0.4 * x + 0.5 * np.sin(2.7 * x) + 1.1 / (1 + x**2)


np.random.seed(5)


def test_IterativeTrimming():
    # generate mock data
    n = 200
    noise = 0.2
    n_outlier = 60
    noise_outlier = 1

    x_ob = np.random.rand(n) * 6 - 3
    y_ob = neal_func(x_ob) + np.random.randn(n) * noise
    y_ob[:n_outlier] = (
        neal_func(x_ob[:n_outlier]) + np.random.randn(n_outlier) * noise_outlier
    )

    experiments = pd.DataFrame()
    experiments["x_1"] = x_ob
    experiments["y"] = y_ob
    experiments["valid_y"] = 1
    inputs = Inputs(
        features=[
            ContinuousInput(
                key=experiments.keys()[0],  # type: ignore
                bounds=(-3, 3),
            )
            for i in range(1)
        ]
    )
    outputs = Outputs(features=[ContinuousOutput(key=experiments.keys()[1])])  # type: ignore
    kernel = ScaleKernel(base_kernel=RBFKernel(ard=True))
    scaler = ScalerEnum.NORMALIZE
    ITGP_model = data_models.IterativeTrimming(
        base_gp=SingleTaskGPSurrogate(
            inputs=inputs, outputs=outputs, kernel=kernel, scaler=scaler
        )
    )
    ITGP = mapper.map(ITGP_model)
    assert isinstance(ITGP.base_gp, SingleTaskGPSurrogate)  # type: ignore
    assert isinstance(ITGP, mapper.IterativeTrimming)
    # detect
    experiments1 = ITGP.detect(experiments=experiments)
    assert len(experiments[experiments["valid_y"] == 1]) != len(
        experiments1[experiments1["valid_y"] == 1]
    )
    assert len(experiments1[experiments1["valid_y"] == 0]) <= n / 2


def test_OutlierDetections():
    # generate mock data
    n = 200
    noise = 0.2
    n_outlier = 60
    noise_outlier = 1
    noise_outlier1 = 1.5
    x_ob = np.random.rand(n) * 6 - 3
    y_ob = neal_func(x_ob) + np.random.randn(n) * noise
    y_ob[:n_outlier] = (
        neal_func(x_ob[:n_outlier]) + np.random.randn(n_outlier) * noise_outlier
    )
    y_ob1 = y_ob
    y_ob1[:n_outlier] = (
        neal_func(x_ob[:n_outlier]) + np.random.randn(n_outlier) * noise_outlier1
    )
    experiments = pd.DataFrame()
    experiments["x_1"] = x_ob
    experiments["y"] = y_ob
    experiments["y1"] = y_ob1
    experiments["valid_y"] = 1
    experiments["valid_y1"] = 1
    inputs = Inputs(
        features=[
            ContinuousInput(
                key=experiments.keys()[0],  # type: ignore
                bounds=(-3, 3),
            )
            for i in range(1)
        ]
    )
    outputs1 = Outputs(features=[ContinuousOutput(key=experiments.keys()[1])])  # type: ignore
    outputs2 = Outputs(features=[ContinuousOutput(key=experiments.keys()[2])])  # type: ignore
    kernel = ScaleKernel(base_kernel=RBFKernel(ard=True))
    scaler = ScalerEnum.NORMALIZE
    ITGP_model1 = data_models.IterativeTrimming(
        base_gp=SingleTaskGPSurrogate(
            inputs=inputs, outputs=outputs1, kernel=kernel, scaler=scaler
        )
    )
    ITGP_model2 = data_models.IterativeTrimming(
        base_gp=SingleTaskGPSurrogate(
            inputs=inputs, outputs=outputs2, kernel=kernel, scaler=scaler
        )
    )
    ITGP = data_models.OutlierDetections(detectors=[ITGP_model1, ITGP_model2])
    ITGP = OutlierDetections(data_model=ITGP)
    assert isinstance(ITGP.detectors[0], mapper.IterativeTrimming)
    assert isinstance(ITGP.detectors[1], mapper.IterativeTrimming)
    # detect
    experiments1 = ITGP.detect(experiments=experiments)
    assert len(experiments[experiments["valid_y"] == 1]) != len(
        experiments1[experiments1["valid_y"] == 1]
    )
    assert len(experiments[experiments["valid_y1"] == 1]) != len(
        experiments1[experiments1["valid_y1"] == 1]
    )
    assert len(experiments1[experiments1["valid_y"] == 0]) <= n / 2
    assert len(experiments1[experiments1["valid_y1"] == 0]) <= n / 2


def test_outlier_detectors_check_compatibility():
    data_model1 = data_models.IterativeTrimming(
        base_gp=SingleTaskGPSurrogate(
            inputs=Inputs(
                features=[
                    ContinuousInput(
                        key=f"x_{i+1}",
                        bounds=(-4, 4),
                    )
                    for i in range(3)
                ]
                + [
                    CategoricalDescriptorInput(
                        key="cat",
                        categories=["apple", "banana"],
                        descriptors=["length", "width"],
                        values=[[1, 2], [3, 4]],
                    )
                ]
            ),
            outputs=Outputs(features=[ContinuousOutput(key="y")]),
            scaler=ScalerEnum.NORMALIZE,
            input_preprocessing_specs={"cat": CategoricalEncodingEnum.ONE_HOT},
        )
    )
    data_model2 = data_models.IterativeTrimming(
        base_gp=SingleTaskGPSurrogate(
            inputs=Inputs(
                features=[
                    ContinuousInput(
                        key=f"x_{i+1}",
                        bounds=(-4, 4),
                    )
                    for i in range(2)
                ]
                + [
                    CategoricalDescriptorInput(
                        key="cat",
                        categories=["apple", "banana"],
                        descriptors=["length", "width"],
                        values=[[1, 2], [3, 4]],
                    )
                ]
            ),
            outputs=Outputs(features=[ContinuousOutput(key="y2")]),
            scaler=ScalerEnum.NORMALIZE,
            input_preprocessing_specs={"cat": CategoricalEncodingEnum.ONE_HOT},
        )
    )
    data_model = data_models.OutlierDetections(detectors=[data_model1, data_model2])
    # models = OutlierDetections(data_model=data_model)
    # check too less input features
    inp = Inputs(
        features=[
            ContinuousInput(
                key=f"x_{i+1}",
                bounds=(-4, 4),
            )
            for i in range(3)
        ]
    )
    out = Outputs(features=[ContinuousOutput(key="y"), ContinuousOutput(key="y2")])
    with pytest.raises(ValueError):
        data_model._check_compability(inp, out)
    # check unused input features
    inp = Inputs(
        features=[
            ContinuousInput(
                key=f"x_{i+1}",
                bounds=(-4, 4),
            )
            for i in range(4)
        ]
        + [
            CategoricalDescriptorInput(
                key="cat",
                categories=["apple", "banana"],
                descriptors=["length", "width"],
                values=[[1, 2], [3, 4]],
            )
        ]
    )
    out = Outputs(features=[ContinuousOutput(key="y"), ContinuousOutput(key="y2")])
    with pytest.raises(ValueError):
        data_model._check_compability(inp, out)
    # check wrong input feature
    inp = Inputs(
        features=[
            ContinuousInput(
                key=f"x_{i+1}",
                bounds=(-4, 4),
            )
            for i in range(4)
        ]
        + [
            ContinuousInput(
                key="cat",
                bounds=(-4, 4),
            )
        ]
    )
    out = Outputs(features=[ContinuousOutput(key="y"), ContinuousOutput(key="y2")])
    with pytest.raises(ValueError):
        data_model._check_compability(inp, out)
    # check too less output features
    inp = Inputs(
        features=[
            ContinuousInput(
                key=f"x_{i+1}",
                bounds=(-4, 4),
            )
            for i in range(3)
        ]
        + [
            CategoricalDescriptorInput(
                key="cat",
                categories=["apple", "banana"],
                descriptors=["length", "width"],
                values=[[1, 2], [3, 4]],
            )
        ]
    )
    out = Outputs(features=[ContinuousOutput(key="y")])
    with pytest.raises(ValueError):
        data_model._check_compability(inp, out)
    # check too many output features
    out = Outputs(
        features=[
            ContinuousOutput(key="y"),
            ContinuousOutput(key="y2"),
            ContinuousOutput(key="y3"),
        ]
    )
    with pytest.raises(ValueError):
        data_model._check_compability(inp, out)
    # check wrong output features
    out = Outputs(
        features=[
            ContinuousOutput(key="y"),
            ContinuousOutput(key="y3"),
        ]
    )
    with pytest.raises(ValueError):
        data_model._check_compability(inp, out)
    # check valid
    inp = Inputs(
        features=[
            ContinuousInput(
                key=f"x_{i+1}",
                bounds=(-4, 4),
            )
            for i in range(3)
        ]
        + [
            CategoricalDescriptorInput(
                key="cat",
                categories=["apple", "banana"],
                descriptors=["length", "width"],
                values=[[1, 2], [3, 4]],
            )
        ]
    )
    out = Outputs(features=[ContinuousOutput(key="y"), ContinuousOutput(key="y2")])
    data_model._check_compability(inp, out)
