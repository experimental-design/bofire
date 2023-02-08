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
)
from torch import Tensor

from bofire.domain.features import (
    CategoricalDescriptorInput,
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
    InputFeatures,
    OutputFeatures,
)
from bofire.models.gps import MixedSingleTaskGPModel, SingleTaskGPModel
from bofire.models.torch_models import BotorchModels, EmpiricalModel
from bofire.utils.enum import CategoricalEncodingEnum, ScalerEnum
from bofire.utils.torch_tools import OneHotToNumeric, tkwargs


@pytest.mark.parametrize("modelclass", [(SingleTaskGPModel), (MixedSingleTaskGPModel)])
def test_BotorchModel_validate_input_preprocessing_steps(modelclass):
    input_features = InputFeatures(
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
    output_features = OutputFeatures(features=[ContinuousOutput(key="y")])
    model = modelclass(
        input_features=input_features,
        output_features=output_features,
    )
    assert model.input_preprocessing_specs == {
        "x_cat": CategoricalEncodingEnum.ONE_HOT,
        "cat": CategoricalEncodingEnum.DESCRIPTOR,
    }
    # test that it can also handle incomplete specs
    model = modelclass(
        input_features=input_features,
        output_features=output_features,
        input_preprocessing_specs={"x_cat": CategoricalEncodingEnum.ONE_HOT},
    )
    assert model.input_preprocessing_specs == {
        "x_cat": CategoricalEncodingEnum.ONE_HOT,
        "cat": CategoricalEncodingEnum.DESCRIPTOR,
    }


@pytest.mark.parametrize(
    "modelclass, input_preprocessing_specs",
    [
        (
            SingleTaskGPModel,
            {
                "x_cat": CategoricalEncodingEnum.DUMMY,
                "cat": CategoricalEncodingEnum.ORDINAL,
            },
        ),
        (
            MixedSingleTaskGPModel,
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
    input_features = InputFeatures(
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
    output_features = OutputFeatures(features=[ContinuousOutput(key="y")])
    with pytest.raises(ValueError):
        modelclass(
            input_features=input_features,
            output_features=output_features,
            input_preprocessing_specs=input_preprocessing_specs,
        )


def test_BotorchModels_invalid_output_features():
    model1 = SingleTaskGPModel(
        input_features=InputFeatures(
            features=[
                ContinuousInput(key=f"x_{i+1}", lower_bound=-4, upper_bound=4)
                for i in range(3)
            ]
        ),
        output_features=OutputFeatures(features=[ContinuousOutput(key="y")]),
        scaler=ScalerEnum.NORMALIZE,
    )
    model2 = SingleTaskGPModel(
        input_features=InputFeatures(
            features=[
                ContinuousInput(key=f"x_{i+1}", lower_bound=-4, upper_bound=4)
                for i in range(2)
            ]
        ),
        output_features=OutputFeatures(features=[ContinuousOutput(key="y")]),
        scaler=ScalerEnum.NORMALIZE,
    )
    with pytest.raises(ValueError):
        BotorchModels(models=[model1, model2])


def test_BotorchModels_invalid_input_features():
    model1 = SingleTaskGPModel(
        input_features=InputFeatures(
            features=[
                ContinuousInput(key=f"x_{i+1}", lower_bound=-4, upper_bound=4)
                for i in range(3)
            ]
        ),
        output_features=OutputFeatures(features=[ContinuousOutput(key="y")]),
        scaler=ScalerEnum.NORMALIZE,
    )
    model2 = SingleTaskGPModel(
        input_features=InputFeatures(
            features=[
                ContinuousInput(key=f"x_{i+1}", lower_bound=-4, upper_bound=4)
                for i in range(2)
            ]
            + [CategoricalInput(key="x_3", categories=["apple", "banana"])]
        ),
        output_features=OutputFeatures(features=[ContinuousOutput(key="y")]),
        scaler=ScalerEnum.NORMALIZE,
    )
    with pytest.raises(ValueError):
        BotorchModels(models=[model1, model2])


def test_BotorchModels_invalid_preprocessing():
    model1 = SingleTaskGPModel(
        input_features=InputFeatures(
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
        output_features=OutputFeatures(features=[ContinuousOutput(key="y")]),
        scaler=ScalerEnum.NORMALIZE,
        input_preprocessing_specs={"cat": CategoricalEncodingEnum.ONE_HOT},
    )
    model2 = SingleTaskGPModel(
        input_features=InputFeatures(
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
        output_features=OutputFeatures(features=[ContinuousOutput(key="y2")]),
        scaler=ScalerEnum.NORMALIZE,
        input_preprocessing_specs={"cat": CategoricalEncodingEnum.DESCRIPTOR},
    )
    with pytest.raises(ValueError):
        BotorchModels(models=[model1, model2])


@pytest.mark.parametrize(
    "models",
    [
        (
            [
                SingleTaskGPModel(
                    input_features=InputFeatures(
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
                    output_features=OutputFeatures(
                        features=[ContinuousOutput(key="y")]
                    ),
                    scaler=ScalerEnum.NORMALIZE,
                    input_preprocessing_specs={"cat": CategoricalEncodingEnum.ONE_HOT},
                ),
                SingleTaskGPModel(
                    input_features=InputFeatures(
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
                    output_features=OutputFeatures(
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
def test_botorch_models_invalid_number_of_outputs(models):
    with pytest.raises(ValueError):
        BotorchModels(models=models)


@pytest.mark.parametrize(
    "models",
    [
        (
            [
                SingleTaskGPModel(
                    input_features=InputFeatures(
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
                    output_features=OutputFeatures(
                        features=[ContinuousOutput(key="y")]
                    ),
                    scaler=ScalerEnum.NORMALIZE,
                    input_preprocessing_specs={"cat": CategoricalEncodingEnum.ONE_HOT},
                ),
                SingleTaskGPModel(
                    input_features=InputFeatures(
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
                    output_features=OutputFeatures(
                        features=[ContinuousOutput(key="y2")]
                    ),
                    scaler=ScalerEnum.NORMALIZE,
                    input_preprocessing_specs={"cat": CategoricalEncodingEnum.ONE_HOT},
                ),
            ]
        )
    ],
)
def test_botorch_models_valid(models):
    BotorchModels(models=models)


def test_botorch_models_check_compatibility():
    model1 = SingleTaskGPModel(
        input_features=InputFeatures(
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
        output_features=OutputFeatures(features=[ContinuousOutput(key="y")]),
        scaler=ScalerEnum.NORMALIZE,
        input_preprocessing_specs={"cat": CategoricalEncodingEnum.ONE_HOT},
    )
    model2 = SingleTaskGPModel(
        input_features=InputFeatures(
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
        output_features=OutputFeatures(features=[ContinuousOutput(key="y2")]),
        scaler=ScalerEnum.NORMALIZE,
        input_preprocessing_specs={"cat": CategoricalEncodingEnum.ONE_HOT},
    )
    models = BotorchModels(models=[model1, model2])
    # check too less input features
    inp = InputFeatures(
        features=[
            ContinuousInput(key=f"x_{i+1}", lower_bound=-4, upper_bound=4)
            for i in range(3)
        ]
    )
    out = OutputFeatures(
        features=[ContinuousOutput(key="y"), ContinuousOutput(key="y2")]
    )
    with pytest.raises(ValueError):
        models._check_compability(inp, out)
    # check unused input features
    inp = InputFeatures(
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
    out = OutputFeatures(
        features=[ContinuousOutput(key="y"), ContinuousOutput(key="y2")]
    )
    with pytest.raises(ValueError):
        models._check_compability(inp, out)
    # check wrong input feature
    inp = InputFeatures(
        features=[
            ContinuousInput(key=f"x_{i+1}", lower_bound=-4, upper_bound=4)
            for i in range(4)
        ]
        + [ContinuousInput(key="cat", lower_bound=-4, upper_bound=4)]
    )
    out = OutputFeatures(
        features=[ContinuousOutput(key="y"), ContinuousOutput(key="y2")]
    )
    with pytest.raises(ValueError):
        models._check_compability(inp, out)
    # check too less output features
    inp = InputFeatures(
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
    out = OutputFeatures(features=[ContinuousOutput(key="y")])
    with pytest.raises(ValueError):
        models._check_compability(inp, out)
    # check too many output features
    out = OutputFeatures(
        features=[
            ContinuousOutput(key="y"),
            ContinuousOutput(key="y2"),
            ContinuousOutput(key="y3"),
        ]
    )
    with pytest.raises(ValueError):
        models._check_compability(inp, out)
    # check wrong output features
    out = OutputFeatures(
        features=[
            ContinuousOutput(key="y"),
            ContinuousOutput(key="y3"),
        ]
    )
    with pytest.raises(ValueError):
        models._check_compability(inp, out)
    # check valid
    inp = InputFeatures(
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
    out = OutputFeatures(
        features=[ContinuousOutput(key="y"), ContinuousOutput(key="y2")]
    )
    models._check_compability(inp, out)


def test_botorch_models_input_preprocessing_specs():
    model1 = SingleTaskGPModel(
        input_features=InputFeatures(
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
        output_features=OutputFeatures(features=[ContinuousOutput(key="y")]),
        scaler=ScalerEnum.NORMALIZE,
        input_preprocessing_specs={"cat": CategoricalEncodingEnum.DESCRIPTOR},
    )
    model2 = SingleTaskGPModel(
        input_features=InputFeatures(
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
        output_features=OutputFeatures(features=[ContinuousOutput(key="y2")]),
        scaler=ScalerEnum.NORMALIZE,
        input_preprocessing_specs={"cat2": CategoricalEncodingEnum.ONE_HOT},
    )
    models = BotorchModels(models=[model1, model2])
    assert models.input_preprocessing_specs == {
        "cat": CategoricalEncodingEnum.DESCRIPTOR,
        "cat2": CategoricalEncodingEnum.ONE_HOT,
    }


def test_botorch_models_fit_and_compatibilize():
    # model 1
    input_features = InputFeatures(
        features=[
            ContinuousInput(key=f"x_{i+1}", lower_bound=-4, upper_bound=4)
            for i in range(2)
        ]
    )
    output_features = OutputFeatures(features=[ContinuousOutput(key="y")])
    experiments1 = input_features.sample(n=10)
    experiments1.eval("y=((x_1**2 + x_2 - 11)**2+(x_1 + x_2**2 -7)**2)", inplace=True)
    experiments1["valid_y"] = 1
    model1 = SingleTaskGPModel(
        input_features=input_features,
        output_features=output_features,
        scaler=ScalerEnum.NORMALIZE,
    )
    # model 2
    input_features = InputFeatures(
        features=[
            ContinuousInput(key=f"x_{i+1}", lower_bound=-4, upper_bound=4)
            for i in range(2)
        ]
        + [CategoricalInput(key="x_cat", categories=["mama", "papa"])]
    )
    output_features = OutputFeatures(features=[ContinuousOutput(key="y2")])
    experiments2 = pd.concat(
        [experiments1, input_features.get_by_key("x_cat").sample(10)], axis=1
    )
    experiments2.eval("y2=((x_1**2 + x_2 - 11)**2+(x_1 + x_2**2 -7)**2)", inplace=True)
    experiments2.loc[experiments2.x_cat == "mama", "y2"] *= 5.0
    experiments2.loc[experiments2.x_cat == "papa", "y2"] /= 2.0
    experiments2["valid_y2"] = 1
    model2 = MixedSingleTaskGPModel(
        input_features=input_features,
        output_features=output_features,
        input_preprocessing_specs={"x_cat": CategoricalEncodingEnum.ONE_HOT},
        scaler=ScalerEnum.STANDARDIZE,
    )
    # create models
    models = BotorchModels(models=[model1, model2])
    # unite experiments
    experiments = pd.concat(
        [experiments1, experiments2[["x_cat", "y2", "valid_y2"]]],
        axis=1,
        ignore_index=False,
    )
    # fit the models
    models.fit(experiments=experiments)
    # make and store predictions for later comparison
    preds1 = model1.predict(experiments1)
    preds2 = model2.predict(experiments2)
    # make compatible
    input_features = InputFeatures(
        features=[
            ContinuousInput(key=f"x_{i+1}", lower_bound=-4, upper_bound=4)
            for i in range(2)
        ]
        + [CategoricalInput(key="x_cat", categories=["mama", "papa"])]
    )
    output_features = OutputFeatures(
        features=[ContinuousOutput(key="y"), ContinuousOutput(key="y2")]
    )
    combined = models.compatibilize(
        input_features=input_features, output_features=output_features
    )
    # check combined
    assert isinstance(combined.models[0], SingleTaskGP)
    assert isinstance(combined.models[1], MixedSingleTaskGP)
    assert isinstance(combined.models[0].input_transform, ChainedInputTransform)
    assert isinstance(combined.models[0].input_transform.tf1, FilterFeatures)
    assert torch.eq(
        combined.models[0].input_transform.tf1.feature_indices,
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
    input_features = InputFeatures(
        features=[
            ContinuousInput(key=f"x_{i+1}", lower_bound=-4, upper_bound=4)
            for i in range(2)
        ]
    )
    output_features = OutputFeatures(features=[ContinuousOutput(key="y")])
    experiments1 = input_features.sample(n=10)
    experiments1.eval("y=((x_1**2 + x_2 - 11)**2+(x_1 + x_2**2 -7)**2)", inplace=True)
    experiments1["valid_y"] = 1
    model1 = EmpiricalModel(
        input_features=input_features, output_features=output_features
    )
    model1.model = HimmelblauModel()
    # test prediction
    preds1 = model1.predict(experiments1)
    assert np.allclose(experiments1.y.values, preds1.y_pred.values)
    # test usage of empirical model in a modellist
    input_features = InputFeatures(
        features=[
            ContinuousInput(key=f"x_{i+1}", lower_bound=-4, upper_bound=4)
            for i in range(2)
        ]
        + [CategoricalInput(key="x_cat", categories=["mama", "papa"])]
    )
    output_features = OutputFeatures(features=[ContinuousOutput(key="y2")])
    experiments2 = pd.concat(
        [experiments1, input_features.get_by_key("x_cat").sample(10)], axis=1
    )
    experiments2.eval("y2=((x_1**2 + x_2 - 11)**2+(x_1 + x_2**2 -7)**2)", inplace=True)
    experiments2.loc[experiments2.x_cat == "mama", "y2"] *= 5.0
    experiments2.loc[experiments2.x_cat == "papa", "y2"] /= 2.0
    experiments2["valid_y2"] = 1
    model2 = MixedSingleTaskGPModel(
        input_features=input_features,
        output_features=output_features,
        input_preprocessing_specs={"x_cat": CategoricalEncodingEnum.ONE_HOT},
        scaler=ScalerEnum.STANDARDIZE,
    )
    # create models
    models = BotorchModels(models=[model1, model2])
    # unite experiments
    experiments = pd.concat(
        [experiments1, experiments2[["x_cat", "y2", "valid_y2"]]],
        axis=1,
        ignore_index=False,
    )
    # fit the models
    models.fit(experiments=experiments)
    # make and store predictions for later comparison
    preds1 = model1.predict(experiments1)
    preds2 = model2.predict(experiments2)
    # make compatible
    input_features = InputFeatures(
        features=[
            ContinuousInput(key=f"x_{i+1}", lower_bound=-4, upper_bound=4)
            for i in range(2)
        ]
        + [CategoricalInput(key="x_cat", categories=["mama", "papa"])]
    )
    output_features = OutputFeatures(
        features=[ContinuousOutput(key="y"), ContinuousOutput(key="y2")]
    )
    combined = models.compatibilize(
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
