from typing import TYPE_CHECKING, Tuple

from pydantic import model_validator

from bofire.data_models.base import BaseModel
from bofire.data_models.feature_context import FeatureContext


if TYPE_CHECKING:
    from bofire.data_models.domain.api import EngineeredFeatures, Inputs
    from bofire.data_models.types import InputTransformSpecs


class KernelBasedSurrogate(BaseModel):
    """Surrogate whose covariance function is chosen, not fixed by the model it builds.

    At construction, every model component it carries is checked against the features
    it is applied to, so a kernel naming an unknown feature or one it cannot work on is
    rejected before any fitting.
    """

    if TYPE_CHECKING:
        # declared by the surrogates this is mixed into
        inputs: "Inputs"
        categorical_encodings: "InputTransformSpecs"
        engineered_features: "EngineeredFeatures"

    @classmethod
    def component_field_names(cls) -> Tuple[str, ...]:
        """Names of the fields holding a kernel, mean or likelihood."""
        return ("kernel",)

    def offered_features(self) -> Tuple[str, ...]:
        """Keys of the features a component receives when it selects none itself."""
        return tuple(self.inputs.get_keys() + self.engineered_features.get_keys())

    @model_validator(mode="after")
    def validate_components(self):
        context = FeatureContext(
            inputs=self.inputs,
            encodings=self.categorical_encodings,
            engineered_features=self.engineered_features,
            offered=self.offered_features(),
        )
        for name in type(self).component_field_names():
            getattr(self, name).validate_inputs(context)
        return self
