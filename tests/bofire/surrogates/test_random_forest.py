import numpy as np
import pytest
import torch
from pandas.testing import assert_frame_equal
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import NotFittedError

import bofire.surrogates.api as surrogates
from bofire.benchmarks.single import Himmelblau
from bofire.data_models.domain.api import Inputs, Outputs
from bofire.data_models.enum import CategoricalEncodingEnum
from bofire.data_models.features.api import (
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
)
from bofire.data_models.surrogates.api import RandomForestSurrogate
from bofire.surrogates.random_forest import _RandomForest


def test_random_forest_no_random_forest_regressor():
    with pytest.raises(ValueError):
        _RandomForest(rf=5)


def test_random_forest_not_fitted():
    with pytest.raises(NotFittedError):
        _RandomForest(rf=RandomForestRegressor())


def test_random_forest_forward():
    bench = Himmelblau()
    samples = bench.domain.inputs.sample(10)
    experiments = bench.f(samples, return_complete=True)
    rfr = RandomForestRegressor().fit(
        experiments[["x_1", "x_2"]].values, experiments.y.values.ravel()
    )
    rf = _RandomForest(rf=rfr)
    pred = rf.forward(torch.from_numpy(experiments[["x_1", "x_2"]].values))
    assert np.allclose(
        rfr.predict(experiments[["x_1", "x_2"]].values),
        pred.numpy().mean(axis=-3).ravel(),
    )
    assert np.allclose(
        rfr.predict(experiments[["x_1", "x_2"]].values),
        rf.posterior(torch.from_numpy(experiments[["x_1", "x_2"]].values))
        .mean.numpy()
        .ravel(),
    )
    assert pred.shape == torch.Size((100, 10, 1))
    # test with batches
    batch = torch.from_numpy(experiments[["x_1", "x_2"]].values).unsqueeze(0)
    pred = rf.forward(batch)
    assert pred.shape == torch.Size((1, 100, 10, 1))
    assert rf.num_outputs == 1


def test_random_forest():
    # test only continuous
    bench = Himmelblau()
    samples = bench.domain.inputs.sample(10)
    experiments = bench.f(samples, return_complete=True)
    rf = RandomForestSurrogate(inputs=bench.domain.inputs, outputs=bench.domain.outputs)
    rf = surrogates.map(rf)
    rf.fit(experiments=experiments)
    # test with categoricals
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
    rf = RandomForestSurrogate(inputs=inputs, outputs=outputs)
    rf = surrogates.map(rf)
    assert rf.input_preprocessing_specs["x_cat"] == CategoricalEncodingEnum.ONE_HOT
    with pytest.raises(ValueError):
        rf.dumps()
    rf.fit(experiments=experiments)
    # test dumps and load
    preds = rf.predict(experiments)
    dump = rf.dumps()
    rf2 = RandomForestSurrogate(inputs=inputs, outputs=outputs)
    rf2 = surrogates.map(rf2)
    rf2.loads(dump)
    preds2 = rf.predict(experiments)
    assert_frame_equal(preds, preds2)
