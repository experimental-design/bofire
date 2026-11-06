import torch
from botorch.models.deterministic import AffineDeterministicModel
from botorch.models.transforms.input import ChainedInputTransform
from typing_extensions import Self

from bofire.data_models.domain.api import EngineeredFeatures, Inputs, Outputs
from bofire.data_models.encodings.api import OneHotEncoding
from bofire.data_models.surrogates.api import (
    CategoricalDeterministicSurrogate as CategoricalDeterministicSurrogateDataModel,
)
from bofire.data_models.surrogates.api import (
    LinearDeterministicSurrogate as LinearDeterministicSurrogateDataModel,
)
from bofire.data_models.types import InputTransformSpecs
from bofire.surrogates.botorch import BotorchSurrogate
from bofire.surrogates.engineered_features import map as map_feature
from bofire.surrogates.surrogate import make_surrogate
from bofire.utils.torch_tools import get_NumericToCategorical_input_transform, tkwargs


class LinearDeterministicSurrogate(BotorchSurrogate):
    @classmethod
    def make(
        cls,
        inputs: Inputs,
        outputs: Outputs,
        coefficients: dict[str, float],
        intercept: float,
        dump: str | None = None,
        categorical_encodings: InputTransformSpecs = {},  # noqa: B006 - copied by make_surrogate
        engineered_features: EngineeredFeatures = EngineeredFeatures(),  # noqa: B008 - copied by make_surrogate
    ) -> Self:
        """Create a deterministic linear model ready for prediction.

        Args:
            inputs: Numeric input features.
            outputs: Output features.
            coefficients: Coefficients keyed by input and engineered feature names.
            intercept: Constant offset.
            dump: Serialized model, if provided.
            categorical_encodings: In-model categorical encodings.
            engineered_features: Features derived from the inputs.

        Defaults mirror the data model and are copied for each call.
        """
        return make_surrogate(cls, LinearDeterministicSurrogateDataModel, locals())

    def __init__(
        self,
        data_model: LinearDeterministicSurrogateDataModel,
        **kwargs,
    ):
        self.intercept = data_model.intercept
        self.coefficients = data_model.coefficients
        super().__init__(data_model=data_model, **kwargs)
        self.model = AffineDeterministicModel(
            b=data_model.intercept,
            a=torch.tensor(
                [data_model.coefficients[key] for key in self.inputs.get_keys()]
                + [
                    data_model.coefficients[key]
                    for key in self.engineered_features.get_keys()
                ],
            )
            .to(**tkwargs)
            .unsqueeze(-1),
        )
        # we have to set an input transform here to do the feature engineering
        if len(self.engineered_features) > 0:
            transforms = {}
            for feature in self.engineered_features.get():
                transforms[feature.key] = map_feature(
                    data_model=feature,
                    inputs=self.inputs,
                    transform_specs=self.categorical_encodings,
                )
            if len(transforms) == 1:
                self.model.input_transform = list(transforms.values())[0]
            else:
                self.model.input_transform = ChainedInputTransform(**transforms)


class CategoricalDeterministicSurrogate(BotorchSurrogate):
    @classmethod
    def make(
        cls,
        inputs: Inputs,
        outputs: Outputs,
        mapping: dict[str, float],
        dump: str | None = None,
        categorical_encodings: InputTransformSpecs = {},  # noqa: B006 - copied by make_surrogate
        engineered_features: EngineeredFeatures = EngineeredFeatures(),  # noqa: B008 - copied by make_surrogate
    ) -> Self:
        """Create a deterministic categorical model ready for prediction.

        Args:
            inputs: Exactly one categorical input feature.
            outputs: Output features.
            mapping: Continuous value for each input category.
            dump: Serialized model, if provided.
            categorical_encodings: In-model categorical encodings.
            engineered_features: Features derived from the inputs.

        Defaults mirror the data model and are copied for each call.
        """
        return make_surrogate(cls, CategoricalDeterministicSurrogateDataModel, locals())

    def __init__(
        self,
        data_model: CategoricalDeterministicSurrogateDataModel,
        **kwargs,
    ):
        self.mapping = data_model.mapping
        super().__init__(data_model=data_model, **kwargs)
        self.model = AffineDeterministicModel(
            b=0.0,
            a=torch.tensor(
                [data_model.mapping[key] for key in self.inputs[0].categories],
            )
            .to(**tkwargs)
            .unsqueeze(-1),
        )
        # as bofire always assumes ordinal encoding for categoricals, we have to map them here explicitly
        # to one hot with a specific input transform
        self.model.input_transform = get_NumericToCategorical_input_transform(  # ty: ignore[unresolved-attribute]
            inputs=self.inputs,
            transform_specs={self.inputs[0].key: OneHotEncoding()},
        )
