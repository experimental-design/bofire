import warnings
from typing import Tuple

import numpy as np
import pandas as pd

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

    def _get_params_dict(self):
        return {
            "tree_train_params": {
                "train_lib": self.train_lib,
                "train_params": {
                    "objective": self.objective,
                    "metric": self.metric,
                    "boosting": self.boosting,
                    "num_boost_round": self.num_boost_round,
                    "max_depth": self.max_depth,
                    "min_data_in_leaf": self.min_data_in_leaf,
                    "min_data_per_group": self.min_data_per_group,
                },
                "unc_params": {
                    "beta": self.beta,
                    "acq_sense": self.acq_sense,
                    "dist_trafo": self.dist_trafo,
                    "dist_metric": self.dist_metric,
                    "cat_metric": self.cat_metric,
                },
            }
        }

    def _fit(self, X: pd.DataFrame, Y: pd.DataFrame, **kwargs):
        transformed_X = self.inputs.transform(X, self.input_preprocessing_specs)
        self._get_params_dict()
        self.model = Enting()
        self.model.fit(X=transformed_X.values, y=Y.values)

    def _predict(self, transformed_X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        preds = self.model.predict(transformed_X.to_numpy())
        # pred has shape [([mu1], std1), ([mu2], std2), ... ]
        m, v = zip(*preds)
        mean = np.array(m)
        std = np.sqrt(np.array(v)).reshape(-1, 1)
        # std is given combined - copy for each objective
        std = np.tile(std, mean.shape[1])
        return mean, std

    def loads(self, data: str):
        pass

    def _dumps(self) -> str:
        pass
