from dataclasses import dataclass
from typing import List, Optional, Tuple

from bofire.data_models.domain.api import EngineeredFeatures, Inputs
from bofire.data_models.encodings.api import AnyCategoricalEncoding
from bofire.data_models.features.api import AnyFeature
from bofire.data_models.types import InputTransformSpecs


@dataclass(frozen=True)
class FeatureContext:
    """What a model component is applied to: the features on offer and their encoding.

    Attributes:
        inputs: The inputs of the surrogate the component belongs to.
        encodings: How each categorical input is encoded, with defaults already filled
            in.
        engineered_features: Quantities the surrogate derives from its inputs.
        offered: Keys of the features a component receives when it leaves `features`
            unset. This is decided by the surrogate, not by the component.
    """

    inputs: Inputs
    encodings: InputTransformSpecs
    engineered_features: EngineeredFeatures
    offered: Tuple[str, ...]

    def keys(self) -> List[str]:
        """Keys of every input and engineered feature."""
        return self.inputs.get_keys() + self.engineered_features.get_keys()

    def get(self, key: str) -> AnyFeature:
        """The input or engineered feature with this key.

        Raises:
            KeyError: If no input or engineered feature has this key.
        """
        if key in self.inputs.get_keys():
            return self.inputs.get_by_key(key)
        if key in self.engineered_features.get_keys():
            return self.engineered_features.get_by_key(key)
        raise KeyError(key)

    def encoding(self, key: str) -> Optional[AnyCategoricalEncoding]:
        """How the feature with this key is encoded; `None` for non-categoricals."""
        return self.encodings.get(key)
