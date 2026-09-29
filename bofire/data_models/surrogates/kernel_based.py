from typing import TYPE_CHECKING, Any, Dict, List, Tuple

from pydantic import model_validator

from bofire.data_models.base import BaseModel
from bofire.data_models.feature_context import FeatureContext


if TYPE_CHECKING:
    from bofire.data_models.domain.api import EngineeredFeatures, Inputs
    from bofire.data_models.encodings.api import AnyCategoricalEncoding
    from bofire.data_models.kernels.api import AnyKernel
    from bofire.data_models.types import InputTransformSpecs


class KernelBasedSurrogate(BaseModel):
    """Surrogate whose covariance function is chosen, not fixed by the model it builds.

    At construction, every model component it carries is checked against the features
    it is applied to, so a kernel naming an unknown feature or one it cannot work on is
    rejected before any fitting. Categoricals left without an encoding get the one the
    components ask for.
    """

    if TYPE_CHECKING:
        # declared by the surrogates this is mixed into
        inputs: "Inputs"
        categorical_encodings: "InputTransformSpecs"
        engineered_features: "EngineeredFeatures"

        def _fill_categorical_encodings(self) -> None: ...

        # all but the mixed GP, which overrides components()
        @property
        def kernel(self) -> "AnyKernel": ...

    def components(self) -> List[Any]:
        """The kernel, mean and likelihood this surrogate is built from."""
        return [self.kernel]

    def offered_features(self) -> Tuple[str, ...]:
        """Keys of the features a component receives when it selects none itself."""
        return tuple(self.inputs.get_keys() + self.engineered_features.get_keys())

    def encoding_requests(self) -> Dict[str, "AnyCategoricalEncoding"]:
        """The encodings the components ask for, by feature key.

        Raises:
            ValueError: If two components ask for different encodings of one feature.
        """
        requests: Dict[str, AnyCategoricalEncoding] = {}
        for component in self.components():
            for key, encoding in component.encoding_requests(self.inputs).items():
                if key in requests and requests[key] != encoding:
                    raise ValueError(
                        f"The components ask for different encodings of '{key}': "
                        f"{type(requests[key]).__name__} and {type(encoding).__name__}."
                    )
                requests[key] = encoding
        return requests

    @model_validator(mode="after")
    def validate_components(self):
        self._fill_categorical_encodings()
        context = FeatureContext(
            inputs=self.inputs,
            encodings=self.categorical_encodings,
            engineered_features=self.engineered_features,
            offered=self.offered_features(),
        )
        for component in self.components():
            component.validate_inputs(context)
        return self
