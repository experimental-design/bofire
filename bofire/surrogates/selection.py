import json
from typing import Optional, cast

import numpy as np
import pandas as pd

from bofire.data_models.enum import REGRESSION_METRIC_DIRECTIONS
from bofire.data_models.surrogates.api import SelectionSurrogate as DataModel
from bofire.surrogates.botorch import TrainableBotorchSurrogate
from bofire.surrogates.surrogate import Surrogate
from bofire.surrogates.trainable import TrainableSurrogate


class SelectionSurrogate(Surrogate, TrainableSurrogate):
    """Chooses one of several surrogate options by cross-validation.

    After fitting, `selected` is the position of the chosen option, `chosen` the
    fitted option that makes the predictions, and `scores` holds the
    cross-validation metrics of every option, one row per option in their order.
    """

    def __init__(
        self,
        data_model: DataModel,
        **kwargs,
    ):
        self.options = data_model.options
        self.metric = data_model.metric
        self.folds = data_model.folds
        # drawn once if not given, so that every option is scored on the same folds
        self.random_state: int = (
            data_model.random_state
            if data_model.random_state is not None
            else np.random.SeedSequence().generate_state(1, dtype=np.uint32).item()
        )
        self.selected: Optional[int] = None
        self.chosen: Optional[TrainableBotorchSurrogate] = None
        self.scores: Optional[pd.DataFrame] = None
        super().__init__(data_model=data_model)

    def _fit(self, X: pd.DataFrame, Y: pd.DataFrame, **kwargs):
        experiments = self.outputs.add_valid_columns(pd.concat([X, Y], axis=1))
        self.selected = self._select(experiments)
        chosen = self._map(self.selected)
        chosen.fit(experiments)
        self.chosen = chosen
        self.model = chosen.model

    def _select(self, experiments: pd.DataFrame) -> int:
        scores = []
        for i in range(len(self.options)):
            _, cv_test, _ = self._map(i).cross_validate(
                experiments,
                folds=self.folds,
                random_state=self.random_state,
            )
            scores.append(cv_test.get_metrics(combine_folds=True).iloc[0])
        self.scores = pd.DataFrame(scores).reset_index(drop=True)
        ranked = self.scores[self.metric.name]
        if REGRESSION_METRIC_DIRECTIONS[self.metric] == "MINIMIZE":
            return int(ranked.idxmin())
        return int(ranked.idxmax())

    def _map(self, i: int) -> TrainableBotorchSurrogate:
        # imported here, as the mapper imports this module
        from bofire.surrogates.mapper import map as map_surrogate

        # every option type maps to a trainable botorch surrogate
        return cast(TrainableBotorchSurrogate, map_surrogate(self.options[i]))

    def _predict(self, transformed_X: pd.DataFrame):
        assert self.chosen is not None
        return self.chosen._predict(transformed_X)

    def _dumps(self) -> str:
        assert self.chosen is not None
        scores = (
            json.loads(self.scores.to_json(orient="split", double_precision=15))
            if self.scores is not None
            else None
        )
        return json.dumps(
            {
                "selected": self.selected,
                "scores": scores,
                "dump": self.chosen.dumps(),
            }
        )

    def loads(self, data: str):
        loaded = json.loads(data)
        self.selected = loaded["selected"]
        scores = loaded.get("scores")
        if scores is not None:
            self.scores = pd.DataFrame(
                data=scores["data"],
                index=scores["index"],
                columns=scores["columns"],
            )
        chosen = self._map(self.selected)
        chosen.loads(loaded["dump"])
        self.chosen = chosen
        self.model = chosen.model
