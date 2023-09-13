import warnings
from typing import Tuple

import numpy as np
import pandas as pd

from bofire.utils.tmpfile import make_tmpfile

try:
    from entmoot.models.enting import Enting  # type: ignore
    from entmoot.problem_config import ProblemConfig
except ImportError:
    warnings.warn("entmoot not installed, BoFire's `EntingSurrogate` cannot be used.")

import uuid

from bofire.data_models.surrogates.api import EntingSurrogate as DataModel
from bofire.surrogates.surrogate import Surrogate
from bofire.surrogates.trainable import TrainableSurrogate


class EntingSurrogate(TrainableSurrogate, Surrogate):
    def __init__(self, data_model: DataModel, **kwargs) -> None:
        self.train_lib = data_model.train_lib

        self.objective = data_model.objective
        self.metric = data_model.metric
        self.boosting = data_model.boosting
        self.num_boost_round = data_model.num_boost_round
        self.max_depth = data_model.max_depth
        self.min_data_in_leaf = data_model.min_data_in_leaf
        self.min_data_per_group = data_model.min_data_per_group

        self.beta = data_model.beta
        self.acq_sense = data_model.acq_sense
        self.dist_trafo = data_model.dist_trafo
        self.dist_metric = data_model.dist_metric
        self.cat_metric = data_model.cat_metric

        self.tmpfile_name = f"enting_{uuid.uuid4().hex}.json"
        super().__init__(data_model=data_model, **kwargs)

    def _init_params(self):
        return dict(
            tree_train_params=dict(
                train_lib = self.train_lib,
                train_params = dict(
                    objective = self.objective,
                    metric = self.metric,
                    boosting = self.boosting,
                    num_boost_round = self.num_boost_round,
                    max_depth = self.max_depth,
                    min_data_in_leaf = self.min_data_in_leaf,
                    min_data_per_group = self.min_data_per_group,
                ),
            unc_params=dict(
                beta=self.beta,
                acq_sense=self.acq_sense,
                dist_trafo=self.dist_trafo,
                dist_metric=self.dist_metric,
                cat_metric=self.cat_metric,
            )
            ))

    def _fit(self, X: pd.DataFrame, Y: pd.DataFrame, **kwargs):
        transformed_X = self.inputs.transform(X, self.input_preprocessing_specs)
        params = self._init_params
        self.model = Enting()
        self.model.fit(X=transformed_X.values, y=Y.values)

    def _predict(self, transformed_X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        preds = self.model.predict(transformed_X.values)
        return preds.reshape((transformed_X.shape[0], 1)), np.zeros(
            (transformed_X.shape[0], 1)
        )

    def loads(self, data: str):
        with make_tmpfile(name=self.tmpfile_name) as fname:
            # write to file
            self._init_xgb()
            with open(fname, "w") as f:
                f.write(data)
            self.model.load_model(fname)

    def _dumps(self) -> str:
        with make_tmpfile(name=self.tmpfile_name) as fname:
            self.model.save_model(fname=fname)
            with open(fname, "r") as f:
                dump = f.read()
            return dump
