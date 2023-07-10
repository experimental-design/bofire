import numpy as np
import pandas as pd

import bofire.outlier_detection.api as mapper
from bofire.data_models.domain.api import Inputs, Outputs
from bofire.data_models.features.api import (
    ContinuousInput,
    ContinuousOutput,
)
from bofire.data_models.kernels.api import RBFKernel, ScaleKernel
from bofire.data_models.outlier_detection.api import IterativeTrimming
from bofire.data_models.surrogates.api import ScalerEnum, SingleTaskGPSurrogate


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
    ITGP_model = IterativeTrimming(
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
    assert len(experiments1[experiments1["valid_y"] == 0]) > n_outlier / 2
