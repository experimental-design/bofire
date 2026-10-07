import json
import warnings
from typing import Optional, cast

import pandas as pd

from bofire.data_models.enum import RegressionMetricsEnum
from bofire.data_models.surrogates.api import SelectionSurrogate as DataModel
from bofire.surrogates.botorch import TrainableBotorchSurrogate
from bofire.surrogates.surrogate import Surrogate
from bofire.surrogates.trainable import TrainableSurrogate


# metrics for which a lower cross-validation score is better
_LOWER_IS_BETTER = {
    RegressionMetricsEnum.MAE,
    RegressionMetricsEnum.MSD,
    RegressionMetricsEnum.MAPE,
}


class SelectionSurrogate(Surrogate, TrainableSurrogate):
    """Chooses one of several candidate surrogates by cross-validation.

    After fitting, `selected` is the position of the chosen candidate, `chosen` the
    fitted candidate that makes the predictions, and `scores` holds the
    cross-validation metrics of every candidate evaluated in the last choice, indexed
    by position.
    """

    def __init__(
        self,
        data_model: DataModel,
        **kwargs,
    ):
        self.candidates = data_model.candidates
        self.metric = data_model.metric
        self.folds = data_model.folds
        self.random_state = data_model.random_state
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
        scores = {}
        for i in range(len(self.candidates)):
            try:
                _, cv_test, _ = self._map(i).cross_validate(
                    experiments,
                    folds=self.folds,
                    random_state=self.random_state,
                )
            except Exception as e:
                warnings.warn(f"Candidate {i} skipped, it could not be fitted: {e}")
                continue
            scores[i] = cv_test.get_metrics(combine_folds=True).iloc[0]
        self.scores = pd.DataFrame.from_dict(scores, orient="index")
        ranked = self.scores[self.metric.name].dropna() if scores else pd.Series()
        if len(ranked) == 0:
            raise ValueError("None of the candidates could be fitted and scored.")
        if self.metric in _LOWER_IS_BETTER:
            return int(ranked.idxmin())
        return int(ranked.idxmax())

    def _map(self, i: int) -> TrainableBotorchSurrogate:
        # imported here, as the mapper imports this module
        from bofire.surrogates.mapper import map as map_surrogate

        # every candidate type maps to a trainable botorch surrogate
        return cast(TrainableBotorchSurrogate, map_surrogate(self.candidates[i]))

    def _predict(self, transformed_X: pd.DataFrame):
        assert self.chosen is not None
        return self.chosen._predict(transformed_X)

    def _dumps(self) -> str:
        assert self.chosen is not None
        return json.dumps({"selected": self.selected, "dump": self.chosen.dumps()})

    def loads(self, data: str):
        loaded = json.loads(data)
        self.selected = loaded["selected"]
        chosen = self._map(self.selected)
        chosen.loads(loaded["dump"])
        self.chosen = chosen
        self.model = chosen.model
