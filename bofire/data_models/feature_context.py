from dataclasses import dataclass, replace
from typing import List, Optional, Tuple, cast

from bofire.data_models.domain.api import EngineeredFeatures, Inputs
from bofire.data_models.encodings.api import AnyCategoricalEncoding, OrdinalEncoding
from bofire.data_models.features.api import AnyFeature, CategoricalTaskInput
from bofire.data_models.types import InputTransformSpecs


def task_input_key(inputs: Inputs, key: Optional[str] = None) -> Optional[str]:
    """Key of the task input a multi-task component works on, if it can be found.

    Args:
        inputs: The inputs to look in.
        key: Key of the task input. If not given, the single task input.

    Returns:
        The key, or `None` if there is no such task input; validation reports why.
    """
    if key is not None:
        is_task = key in inputs.get_keys() and isinstance(
            inputs.get_by_key(key), CategoricalTaskInput
        )
        return key if is_task else None
    keys = inputs.get_keys(CategoricalTaskInput)
    return keys[0] if len(keys) == 1 else None


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

    def without(self, key: str) -> "FeatureContext":
        """The same context with one feature no longer offered."""
        return replace(self, offered=tuple(k for k in self.offered if k != key))

    def only(self, keys: List[str]) -> "FeatureContext":
        """The same context offering only the given features, in offered order."""
        return replace(self, offered=tuple(k for k in self.offered if k in keys))

    def task_feature(self, key: Optional[str] = None) -> CategoricalTaskInput:
        """The task input a multi-task component works on.

        Args:
            key: Key of the task input. If not given, the single task input on offer.

        Returns:
            The task input.

        Raises:
            ValueError: If there is no such task input, or not exactly one when no key
                is given, or it is not encoded as ordinal codes.
        """
        if key is None:
            candidates = [
                k for k in self.offered if isinstance(self.get(k), CategoricalTaskInput)
            ]
            if len(candidates) != 1:
                raise ValueError(
                    f"Exactly one task input is required, found {candidates}."
                )
            (key,) = candidates
        elif key not in self.keys() or not isinstance(
            self.get(key), CategoricalTaskInput
        ):
            raise ValueError(f"'{key}' is not a task input.")
        if not isinstance(self.encoding(key), OrdinalEncoding):
            raise ValueError(
                f"The task input '{key}' has to be encoded as ordinal codes."
            )
        return cast(CategoricalTaskInput, self.get(key))
