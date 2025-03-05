import importlib

import pytest
from pandas.testing import assert_frame_equal

import bofire.surrogates.api as surrogates
from bofire.benchmarks.single import Himmelblau
from bofire.data_models.domain.api import Inputs, Outputs
from bofire.data_models.enum import CategoricalEncodingEnum
from bofire.data_models.features.api import (
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
)
from bofire.data_models.surrogates.api import XGBoostSurrogate


XGB_AVAILABLE = importlib.util.find_spec("xgboost") is not None


@pytest.mark.skipif(not XGB_AVAILABLE, reason="requires xgboost")
def test_XGBoostSurrogate():
    benchmark = Himmelblau()
    samples = benchmark.domain.inputs.sample(10)
    experiments = benchmark.f(samples, return_complete=True)
    data_model = XGBoostSurrogate(
        inputs=benchmark.domain.inputs,
        outputs=benchmark.domain.outputs,
        n_estimators=2,
    )
    surrogate = surrogates.map(data_model)
    assert isinstance(surrogate, surrogates.XGBoostSurrogate)
    assert surrogate.input_preprocessing_specs == {}
    assert surrogate.is_fitted is False
    # fit it
    surrogate.fit(experiments=experiments)
    assert surrogate.is_fitted is True
    # predict it
    preds = surrogate.predict(experiments)
    # dump it
    dump = surrogate.dumps()
    # load it
    surrogate2 = surrogates.map(data_model)
    surrogate2.loads(dump)
    preds2 = surrogate2.predict(experiments)
    assert_frame_equal(preds, preds2)
    assert_frame_equal(preds, preds2)


@pytest.mark.skipif(not XGB_AVAILABLE, reason="requires xgboost")
def test_XGBoostSurrogate_categorical():
    inputs = Inputs(
        features=[
            ContinuousInput(
                key=f"x_{i+1}",
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
    data_model = XGBoostSurrogate(inputs=inputs, outputs=outputs, n_estimators=2)
    assert data_model.input_preprocessing_specs == {
        "x_cat": CategoricalEncodingEnum.ONE_HOT,
    }
    surrogate = surrogates.map(data_model)
    surrogate.fit(experiments)
