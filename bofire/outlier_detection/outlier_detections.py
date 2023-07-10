import itertools
from abc import ABC
from typing import List

import pandas as pd

from bofire.data_models.domain.api import Inputs, Outputs
from bofire.data_models.outlier_detection.api import OutlierDetections as DataModel
from bofire.outlier_detection.api import (  # noqa: F401
    IterativeTrimming,
    OutlierDetection,
)
from bofire.outlier_detection.api import map as map_outlier

OutlierDetectors = IterativeTrimming


class OutlierDetections(ABC):
    detectors: List[OutlierDetectors]

    def __init__(
        self,
        data_model: DataModel,
        **kwargs,
    ):
        self.detectors = [map_outlier(model) for model in data_model.detectors]  # type: ignore

    def detect(self, experiments: pd.DataFrame) -> pd.DataFrame:
        filtered_experiments = experiments.copy()
        for outlier_model in self.detectors:  # type: ignore
            filtered_experiments = outlier_model.detect(filtered_experiments)

        return filtered_experiments

    @property
    def outputs(self) -> Outputs:
        return Outputs(
            features=list(
                itertools.chain.from_iterable(
                    [model.surrogate.outputs.get() for model in self.detectors]  # type: ignore
                )
            )
        )

    # TODO: is this really neede here, code duplication with functional model
    def _check_compability(self, inputs: Inputs, outputs: Outputs):
        used_output_feature_keys = self.outputs.get_keys()
        if sorted(used_output_feature_keys) != sorted(outputs.get_keys()):
            raise ValueError("Output features do not match.")
        used_feature_keys = []
        for i, model in enumerate(self.detectors):
            if len(model.surrogate.inputs) > len(inputs):  # type: ignore
                raise ValueError(
                    f"Model with index {i} has more features than acceptable."
                )
            for feat in model.surrogate.inputs:  # type: ignore
                try:
                    other_feat = inputs.get_by_key(feat.key)
                except KeyError:
                    raise ValueError(f"Feature {feat.key} not found.")
                # now compare the features
                # TODO: make more sohisticated comparisons based on the type
                # has to to be implemented in features, for the start
                # we go with __eq__
                if feat != other_feat:
                    raise ValueError(f"Features with key {feat.key} are incompatible.")
                if feat.key not in used_feature_keys:
                    used_feature_keys.append(feat.key)
        if len(used_feature_keys) != len(inputs):
            raise ValueError("Unused features are present.")

    def compatibilize(self, inputs: Inputs, outputs: Outputs) -> List[OutlierDetectors]:
        # TODO: add sync option
        # check if models are compatible to provided inputs and outputs
        # of the optimization domain
        self._check_compability(inputs=inputs, outputs=outputs)
        outlier_detection_models = []
        # we sort the models by sorting them with their occurence in outputs
        for output_feature_key in outputs.get_keys():
            # get the corresponding model
            model = {model.surrogate.outputs[0].key: model for model in self.detectors}[  # type: ignore
                output_feature_key
            ]
            if model.surrogate.model is None:
                raise ValueError(
                    f"base_gp for outlier_detector for output feature {output_feature_key} not fitted."
                )

            outlier_detection_models.append(model)

        return outlier_detection_models
