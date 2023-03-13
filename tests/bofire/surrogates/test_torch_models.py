from importlib.util import find_spec

import numpy as np
import pandas as pd
import pytest
import torch
from botorch.models import MixedSingleTaskGP, SingleTaskGP
from botorch.models.deterministic import DeterministicModel
from botorch.models.transforms.input import (
    ChainedInputTransform,
    FilterFeatures,
    InputStandardize,
    Normalize,
    OneHotToNumeric,
)
from pandas.testing import assert_frame_equal
from torch import Tensor

import bofire.data_models.surrogates.api as data_models
import bofire.surrogates.api as surrogates
from bofire.data_models.domain.api import Constraints, Inputs, Outputs
from bofire.data_models.enum import CategoricalEncodingEnum
from bofire.data_models.features.api import (
    CategoricalDescriptorInput,
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
)
from bofire.data_models.surrogates.api import ScalerEnum
from bofire.surrogates.api import BotorchSurrogates
from bofire.surrogates.torch_tools import tkwargs

CLOUDPICKLE_NOT_AVAILABLE = find_spec("cloudpickle") is None


@pytest.mark.parametrize(
    "modelclass",
    [(data_models.SingleTaskGPSurrogate), (data_models.MixedSingleTaskGPSurrogate)],
)
def test_BotorchModel_validate_input_preprocessing_steps(modelclass):
    input_features = Inputs(
        features=[
            ContinuousInput(key=f"x_{i+1}", lower_bound=-4, upper_bound=4)
            for i in range(5)
        ]
        + [
            CategoricalInput(key="x_cat", categories=["mama", "papa"]),
            CategoricalDescriptorInput(
                key="cat",
                categories=["apple", "banana"],
                descriptors=["length", "width"],
                values=[[1, 2], [3, 4]],
            ),
        ]
    )
    output_features = Outputs(features=[ContinuousOutput(key="y")])
    data_model = modelclass(
        input_features=input_features,
        output_features=output_features,
        constraints=Constraints(),
    )
    surrogate = surrogates.map(data_model)
    assert surrogate.input_preprocessing_specs == {
        "x_cat": CategoricalEncodingEnum.ONE_HOT,
        "cat": CategoricalEncodingEnum.DESCRIPTOR,
    }
    # test that it can also handle incomplete specs
    data_model = modelclass(
        input_features=input_features,
        output_features=output_features,
        constraints=Constraints(),
        input_preprocessing_specs={"x_cat": CategoricalEncodingEnum.ONE_HOT},
    )
    surrogate = surrogates.map(data_model)
    assert surrogate.input_preprocessing_specs == {
        "x_cat": CategoricalEncodingEnum.ONE_HOT,
        "cat": CategoricalEncodingEnum.DESCRIPTOR,
    }


@pytest.mark.parametrize(
    "modelclass, input_preprocessing_specs",
    [
        (
            data_models.SingleTaskGPSurrogate,
            {
                "x_cat": CategoricalEncodingEnum.DUMMY,
                "cat": CategoricalEncodingEnum.ORDINAL,
            },
        ),
        (
            data_models.MixedSingleTaskGPSurrogate,
            {
                "x_cat": CategoricalEncodingEnum.ORDINAL,
                "cat": CategoricalEncodingEnum.ONE_HOT,
            },
        ),
    ],
)
def test_BotorchModel_validate_invalid_input_preprocessing_steps(
    modelclass, input_preprocessing_specs
):
    input_features = Inputs(
        features=[
            ContinuousInput(key=f"x_{i+1}", lower_bound=-4, upper_bound=4)
            for i in range(5)
        ]
        + [
            CategoricalInput(key="x_cat", categories=["mama", "papa"]),
            CategoricalDescriptorInput(
                key="cat",
                categories=["apple", "banana"],
                descriptors=["length", "width"],
                values=[[1, 2], [3, 4]],
            ),
        ]
    )
    output_features = Outputs(features=[ContinuousOutput(key="y")])
    with pytest.raises(ValueError):
        modelclass(
            input_features=input_features,
            output_features=output_features,
            input_preprocessing_specs=input_preprocessing_specs,
        )


def test_BotorchSurrogates_invalid_output_features():
    data_model1 = data_models.SingleTaskGPSurrogate(
        input_features=Inputs(
            features=[
                ContinuousInput(key=f"x_{i+1}", lower_bound=-4, upper_bound=4)
                for i in range(3)
            ]
        ),
        output_features=Outputs(features=[ContinuousOutput(key="y")]),
        scaler=ScalerEnum.NORMALIZE,
    )
    data_model2 = data_models.SingleTaskGPSurrogate(
        input_features=Inputs(
            features=[
                ContinuousInput(key=f"x_{i+1}", lower_bound=-4, upper_bound=4)
                for i in range(2)
            ]
        ),
        output_features=Outputs(features=[ContinuousOutput(key="y")]),
        scaler=ScalerEnum.NORMALIZE,
    )
    with pytest.raises(ValueError):
        data_models.BotorchSurrogates(surrogates=[data_model1, data_model2])


def test_BotorchSurrogates_invalid_input_features():
    data_model1 = data_models.SingleTaskGPSurrogate(
        input_features=Inputs(
            features=[
                ContinuousInput(key=f"x_{i+1}", lower_bound=-4, upper_bound=4)
                for i in range(3)
            ]
        ),
        output_features=Outputs(features=[ContinuousOutput(key="y")]),
        scaler=ScalerEnum.NORMALIZE,
    )
    data_model2 = data_models.SingleTaskGPSurrogate(
        input_features=Inputs(
            features=[
                ContinuousInput(key=f"x_{i+1}", lower_bound=-4, upper_bound=4)
                for i in range(2)
            ]
            + [CategoricalInput(key="x_3", categories=["apple", "banana"])]
        ),
        output_features=Outputs(features=[ContinuousOutput(key="y")]),
        scaler=ScalerEnum.NORMALIZE,
    )
    with pytest.raises(ValueError):
        data_models.BotorchSurrogates(surrogates=[data_model1, data_model2])


def test_BotorchSurrogates_invalid_preprocessing():
    data_model1 = data_models.SingleTaskGPSurrogate(
        input_features=Inputs(
            features=[
                ContinuousInput(key=f"x_{i+1}", lower_bound=-4, upper_bound=4)
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
        output_features=Outputs(features=[ContinuousOutput(key="y")]),
        scaler=ScalerEnum.NORMALIZE,
        input_preprocessing_specs={"cat": CategoricalEncodingEnum.ONE_HOT},
    )
    data_model2 = data_models.SingleTaskGPSurrogate(
        input_features=Inputs(
            features=[
                ContinuousInput(key=f"x_{i+1}", lower_bound=-4, upper_bound=4)
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
        output_features=Outputs(features=[ContinuousOutput(key="y2")]),
        scaler=ScalerEnum.NORMALIZE,
        input_preprocessing_specs={"cat": CategoricalEncodingEnum.DESCRIPTOR},
    )
    with pytest.raises(ValueError):
        data_models.BotorchSurrogates(surrogates=[data_model1, data_model2])


@pytest.mark.parametrize(
    "surrogate_list",
    [
        (
            [
                data_models.SingleTaskGPSurrogate(
                    input_features=Inputs(
                        features=[
                            ContinuousInput(
                                key=f"x_{i+1}", lower_bound=-4, upper_bound=4
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
                    output_features=Outputs(features=[ContinuousOutput(key="y")]),
                    scaler=ScalerEnum.NORMALIZE,
                    input_preprocessing_specs={"cat": CategoricalEncodingEnum.ONE_HOT},
                ),
                data_models.SingleTaskGPSurrogate(
                    input_features=Inputs(
                        features=[
                            ContinuousInput(
                                key=f"x_{i+1}", lower_bound=-4, upper_bound=4
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
                    output_features=Outputs(
                        features=[
                            ContinuousOutput(key="y2"),
                            ContinuousOutput(key="y3"),
                        ]
                    ),
                    scaler=ScalerEnum.NORMALIZE,
                    input_preprocessing_specs={"cat": CategoricalEncodingEnum.ONE_HOT},
                ),
            ]
        )
    ],
)
def test_botorch_models_invalid_number_of_outputs(surrogate_list):
    with pytest.raises(ValueError):
        data_models.BotorchSurrogates(surrogates=surrogate_list)


@pytest.mark.parametrize(
    "surrogate_list",
    [
        (
            [
                data_models.SingleTaskGPSurrogate(
                    input_features=Inputs(
                        features=[
                            ContinuousInput(
                                key=f"x_{i+1}", lower_bound=-4, upper_bound=4
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
                    output_features=Outputs(features=[ContinuousOutput(key="y")]),
                    scaler=ScalerEnum.NORMALIZE,
                    input_preprocessing_specs={"cat": CategoricalEncodingEnum.ONE_HOT},
                ),
                data_models.SingleTaskGPSurrogate(
                    input_features=Inputs(
                        features=[
                            ContinuousInput(
                                key=f"x_{i+1}", lower_bound=-4, upper_bound=4
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
                    output_features=Outputs(features=[ContinuousOutput(key="y2")]),
                    scaler=ScalerEnum.NORMALIZE,
                    input_preprocessing_specs={"cat": CategoricalEncodingEnum.ONE_HOT},
                ),
            ]
        )
    ],
)
def test_botorch_models_valid(surrogate_list):
    data_model = data_models.BotorchSurrogates(surrogates=surrogate_list)
    BotorchSurrogates(data_model=data_model)


def test_botorch_models_check_compatibility():
    data_model1 = data_models.SingleTaskGPSurrogate(
        input_features=Inputs(
            features=[
                ContinuousInput(key=f"x_{i+1}", lower_bound=-4, upper_bound=4)
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
        output_features=Outputs(features=[ContinuousOutput(key="y")]),
        scaler=ScalerEnum.NORMALIZE,
        input_preprocessing_specs={"cat": CategoricalEncodingEnum.ONE_HOT},
    )
    data_model2 = data_models.SingleTaskGPSurrogate(
        input_features=Inputs(
            features=[
                ContinuousInput(key=f"x_{i+1}", lower_bound=-4, upper_bound=4)
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
        output_features=Outputs(features=[ContinuousOutput(key="y2")]),
        scaler=ScalerEnum.NORMALIZE,
        input_preprocessing_specs={"cat": CategoricalEncodingEnum.ONE_HOT},
    )
    data_model = data_models.BotorchSurrogates(surrogates=[data_model1, data_model2])
    models = BotorchSurrogates(data_model=data_model)
    # check too less input features
    inp = Inputs(
        features=[
            ContinuousInput(key=f"x_{i+1}", lower_bound=-4, upper_bound=4)
            for i in range(3)
        ]
    )
    out = Outputs(features=[ContinuousOutput(key="y"), ContinuousOutput(key="y2")])
    with pytest.raises(ValueError):
        models._check_compability(inp, out)
    # check unused input features
    inp = Inputs(
        features=[
            ContinuousInput(key=f"x_{i+1}", lower_bound=-4, upper_bound=4)
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
        models._check_compability(inp, out)
    # check wrong input feature
    inp = Inputs(
        features=[
            ContinuousInput(key=f"x_{i+1}", lower_bound=-4, upper_bound=4)
            for i in range(4)
        ]
        + [ContinuousInput(key="cat", lower_bound=-4, upper_bound=4)]
    )
    out = Outputs(features=[ContinuousOutput(key="y"), ContinuousOutput(key="y2")])
    with pytest.raises(ValueError):
        models._check_compability(inp, out)
    # check too less output features
    inp = Inputs(
        features=[
            ContinuousInput(key=f"x_{i+1}", lower_bound=-4, upper_bound=4)
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
        models._check_compability(inp, out)
    # check too many output features
    out = Outputs(
        features=[
            ContinuousOutput(key="y"),
            ContinuousOutput(key="y2"),
            ContinuousOutput(key="y3"),
        ]
    )
    with pytest.raises(ValueError):
        models._check_compability(inp, out)
    # check wrong output features
    out = Outputs(
        features=[
            ContinuousOutput(key="y"),
            ContinuousOutput(key="y3"),
        ]
    )
    with pytest.raises(ValueError):
        models._check_compability(inp, out)
    # check valid
    inp = Inputs(
        features=[
            ContinuousInput(key=f"x_{i+1}", lower_bound=-4, upper_bound=4)
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
    models._check_compability(inp, out)


def test_botorch_models_input_preprocessing_specs():
    data_model1 = data_models.SingleTaskGPSurrogate(
        input_features=Inputs(
            features=[
                ContinuousInput(key=f"x_{i+1}", lower_bound=-4, upper_bound=4)
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
        output_features=Outputs(features=[ContinuousOutput(key="y")]),
        scaler=ScalerEnum.NORMALIZE,
        input_preprocessing_specs={"cat": CategoricalEncodingEnum.DESCRIPTOR},
    )
    data_model2 = data_models.SingleTaskGPSurrogate(
        input_features=Inputs(
            features=[
                ContinuousInput(key=f"x_{i+1}", lower_bound=-4, upper_bound=4)
                for i in range(2)
            ]
            + [
                CategoricalInput(
                    key="cat2",
                    categories=["lotta", "sarah"],
                )
            ]
        ),
        output_features=Outputs(features=[ContinuousOutput(key="y2")]),
        scaler=ScalerEnum.NORMALIZE,
        input_preprocessing_specs={"cat2": CategoricalEncodingEnum.ONE_HOT},
    )
    data_model = data_models.BotorchSurrogates(surrogates=[data_model1, data_model2])
    surrogate = BotorchSurrogates(data_model=data_model)
    assert surrogate.input_preprocessing_specs == {
        "cat": CategoricalEncodingEnum.DESCRIPTOR,
        "cat2": CategoricalEncodingEnum.ONE_HOT,
    }


def test_botorch_models_fit_and_compatibilize():
    # model 1
    input_features = Inputs(
        features=[
            ContinuousInput(key=f"x_{i+1}", lower_bound=-4, upper_bound=4)
            for i in range(2)
        ]
    )
    output_features = Outputs(features=[ContinuousOutput(key="y")])
    experiments1 = input_features.sample(n=10)
    experiments1.eval("y=((x_1**2 + x_2 - 11)**2+(x_1 + x_2**2 -7)**2)", inplace=True)
    experiments1["valid_y"] = 1
    data_model1 = data_models.SingleTaskGPSurrogate(
        input_features=input_features,
        output_features=output_features,
        scaler=ScalerEnum.NORMALIZE,
    )
    # model 2
    input_features = Inputs(
        features=[
            ContinuousInput(key=f"x_{i+1}", lower_bound=-4, upper_bound=4)
            for i in range(2)
        ]
        + [CategoricalInput(key="x_cat", categories=["mama", "papa"])]
    )
    output_features = Outputs(features=[ContinuousOutput(key="y2")])
    experiments2 = pd.concat(
        [experiments1, input_features.get_by_key("x_cat").sample(10)], axis=1
    )
    experiments2.eval("y2=((x_1**2 + x_2 - 11)**2+(x_1 + x_2**2 -7)**2)", inplace=True)
    experiments2.loc[experiments2.x_cat == "mama", "y2"] *= 5.0
    experiments2.loc[experiments2.x_cat == "papa", "y2"] /= 2.0
    experiments2["valid_y2"] = 1
    data_model2 = data_models.MixedSingleTaskGPSurrogate(
        input_features=input_features,
        output_features=output_features,
        input_preprocessing_specs={"x_cat": CategoricalEncodingEnum.ONE_HOT},
        scaler=ScalerEnum.STANDARDIZE,
        constraints=Constraints(),
    )
    # create models
    data_model = data_models.BotorchSurrogates(surrogates=[data_model1, data_model2])
    botorch_surrogates = BotorchSurrogates(data_model=data_model)
    # unite experiments
    experiments = pd.concat(
        [experiments1, experiments2[["x_cat", "y2", "valid_y2"]]],
        axis=1,
        ignore_index=False,
    )
    # fit the models
    botorch_surrogates.fit(experiments=experiments)
    assert botorch_surrogates.surrogates[0].is_compatibilized is False
    assert botorch_surrogates.surrogates[1].is_compatibilized is False
    # make and store predictions for later comparison
    preds1 = botorch_surrogates.surrogates[0].predict(experiments1)
    preds2 = botorch_surrogates.surrogates[1].predict(experiments2)
    # make compatible
    input_features = Inputs(
        features=[
            ContinuousInput(key=f"x_{i+1}", lower_bound=-4, upper_bound=4)
            for i in range(2)
        ]
        + [CategoricalInput(key="x_cat", categories=["mama", "papa"])]
    )
    output_features = Outputs(
        features=[ContinuousOutput(key="y"), ContinuousOutput(key="y2")]
    )
    combined = botorch_surrogates.compatibilize(
        input_features=input_features, output_features=output_features
    )
    assert botorch_surrogates.surrogates[0].is_compatibilized is True
    assert botorch_surrogates.surrogates[1].is_compatibilized is False
    # check combined
    assert isinstance(combined.models[0], SingleTaskGP)
    assert isinstance(combined.models[1], MixedSingleTaskGP)
    assert isinstance(combined.models[0].input_transform, ChainedInputTransform)
    assert isinstance(combined.models[0].input_transform.tcompatibilize, FilterFeatures)
    assert torch.eq(
        combined.models[0].input_transform.tcompatibilize.feature_indices,
        torch.tensor([0, 1], dtype=torch.int64),
    ).all()
    assert isinstance(combined.models[0].input_transform.tf2, Normalize)
    assert isinstance(combined.models[1].input_transform, ChainedInputTransform)
    assert isinstance(combined.models[1].input_transform.tf1, InputStandardize)
    assert isinstance(combined.models[1].input_transform.tf2, OneHotToNumeric)
    # check predictions
    # transform experiments to torch
    trX = input_features.transform(
        experiments=experiments, specs={"x_cat": CategoricalEncodingEnum.ONE_HOT}
    )
    X = torch.from_numpy(trX.values).to(**tkwargs)
    with torch.no_grad():
        preds = combined.posterior(X).mean.detach().numpy()

    assert np.allclose(preds1.y_pred.values, preds[:, 0])
    assert np.allclose(preds2.y2_pred.values, preds[:, 1])
    ## now decompatibilize the models again
    botorch_surrogates.surrogates[0].decompatibilize()
    botorch_surrogates.surrogates[1].decompatibilize()
    assert botorch_surrogates.surrogates[0].is_compatibilized is False
    assert botorch_surrogates.surrogates[1].is_compatibilized is False
    # check again the predictions
    preds11 = botorch_surrogates.surrogates[0].predict(experiments1)
    preds22 = botorch_surrogates.surrogates[1].predict(experiments2)
    assert_frame_equal(preds1, preds11)
    assert_frame_equal(preds2, preds22)
    assert isinstance(botorch_surrogates.surrogates[0].model.input_transform, Normalize)


class HimmelblauModel(DeterministicModel):
    def __init__(self):
        super().__init__()
        self._num_outputs = 1

    def forward(self, X: Tensor) -> Tensor:
        return (
            (X[..., 0] ** 2 + X[..., 1] - 11.0) ** 2
            + (X[..., 0] + X[..., 1] ** 2 - 7.0) ** 2
        ).unsqueeze(-1)


def test_empirical_model():
    input_features = Inputs(
        features=[
            ContinuousInput(key=f"x_{i+1}", lower_bound=-4, upper_bound=4)
            for i in range(2)
        ]
    )
    output_features = Outputs(features=[ContinuousOutput(key="y")])
    experiments1 = input_features.sample(n=10)
    experiments1.eval("y=((x_1**2 + x_2 - 11)**2+(x_1 + x_2**2 -7)**2)", inplace=True)
    experiments1["valid_y"] = 1
    data_model1 = data_models.EmpiricalSurrogate(
        input_features=input_features, output_features=output_features
    )
    surrogate1 = surrogates.map(data_model1)
    surrogate1.model = HimmelblauModel()
    # test prediction
    preds1 = surrogate1.predict(experiments1)
    assert np.allclose(experiments1.y.values, preds1.y_pred.values)
    # test usage of empirical model in a modellist
    input_features = Inputs(
        features=[
            ContinuousInput(key=f"x_{i+1}", lower_bound=-4, upper_bound=4)
            for i in range(2)
        ]
        + [CategoricalInput(key="x_cat", categories=["mama", "papa"])]
    )
    output_features = Outputs(features=[ContinuousOutput(key="y2")])
    experiments2 = pd.concat(
        [experiments1, input_features.get_by_key("x_cat").sample(10)], axis=1
    )
    experiments2.eval("y2=((x_1**2 + x_2 - 11)**2+(x_1 + x_2**2 -7)**2)", inplace=True)
    experiments2.loc[experiments2.x_cat == "mama", "y2"] *= 5.0
    experiments2.loc[experiments2.x_cat == "papa", "y2"] /= 2.0
    experiments2["valid_y2"] = 1
    data_model2 = data_models.MixedSingleTaskGPSurrogate(
        input_features=input_features,
        output_features=output_features,
        input_preprocessing_specs={"x_cat": CategoricalEncodingEnum.ONE_HOT},
        scaler=ScalerEnum.STANDARDIZE,
        constraints=Constraints(),
    )
    # create models
    data_model = data_models.BotorchSurrogates(surrogates=[data_model1, data_model2])
    botorch_surrogates = BotorchSurrogates(data_model=data_model)
    # unite experiments
    experiments = pd.concat(
        [experiments1, experiments2[["x_cat", "y2", "valid_y2"]]],
        axis=1,
        ignore_index=False,
    )
    # fit the models
    botorch_surrogates.fit(experiments=experiments)
    botorch_surrogates.surrogates[0].model = HimmelblauModel()
    # make and store predictions for later comparison
    preds1 = botorch_surrogates.surrogates[0].predict(experiments1)
    preds2 = botorch_surrogates.surrogates[1].predict(experiments2)
    # make compatible
    input_features = Inputs(
        features=[
            ContinuousInput(key=f"x_{i+1}", lower_bound=-4, upper_bound=4)
            for i in range(2)
        ]
        + [CategoricalInput(key="x_cat", categories=["mama", "papa"])]
    )
    output_features = Outputs(
        features=[ContinuousOutput(key="y"), ContinuousOutput(key="y2")]
    )
    combined = botorch_surrogates.compatibilize(
        input_features=input_features, output_features=output_features
    )
    # check combined
    assert isinstance(combined.models[0], DeterministicModel)
    assert isinstance(combined.models[1], MixedSingleTaskGP)
    assert isinstance(combined.models[0].input_transform, FilterFeatures)
    assert torch.eq(
        combined.models[0].input_transform.feature_indices,
        torch.tensor([0, 1], dtype=torch.int64),
    ).all()
    assert isinstance(combined.models[1].input_transform, ChainedInputTransform)
    assert isinstance(combined.models[1].input_transform.tf1, InputStandardize)
    assert isinstance(combined.models[1].input_transform.tf2, OneHotToNumeric)
    # check predictions
    # transform experiments to torch
    trX = input_features.transform(
        experiments=experiments, specs={"x_cat": CategoricalEncodingEnum.ONE_HOT}
    )
    X = torch.from_numpy(trX.values).to(**tkwargs)
    with torch.no_grad():
        preds = combined.posterior(X).mean.detach().numpy()

    assert np.allclose(preds1.y_pred.values, preds[:, 0])
    assert np.allclose(preds2.y2_pred.values, preds[:, 1])


@pytest.mark.skipif(CLOUDPICKLE_NOT_AVAILABLE, reason="requires cloudpickle")
def test_empirical_model_io():
    input_features = Inputs(
        features=[
            ContinuousInput(key=f"x_{i+1}", lower_bound=-4, upper_bound=4)
            for i in range(2)
        ]
    )
    output_features = Outputs(features=[ContinuousOutput(key="y")])
    data_model = data_models.EmpiricalSurrogate(
        input_features=input_features, output_features=output_features
    )
    surrogate = surrogates.map(data_model)
    surrogate.model = HimmelblauModel()
    samples = input_features.sample(5)
    preds = surrogate.predict(samples)
    dump = surrogate.dumps()
    #
    data_model2 = data_models.EmpiricalSurrogate(
        input_features=input_features, output_features=output_features
    )
    surrogate2 = surrogates.map(data_model2)
    surrogate2.loads(dump)
    preds2 = surrogate2.predict(samples)
    assert_frame_equal(preds, preds2)
