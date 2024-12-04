import warnings
from typing import Tuple

import numpy as np
import pandas as pd

from bofire.utils.tmpfile import make_tmpfile


try:
    from xgboost import XGBRegressor
except ImportError:
    warnings.warn(
        "xgboost not installed. Please install it to use "
        "BoFire's `XGBoostSurrogate`.",
        ImportWarning,
    )

import uuid

from bofire.data_models.surrogates.api import XGBoostSurrogate as DataModel
from bofire.surrogates.surrogate import Surrogate
from bofire.surrogates.trainable import TrainableSurrogate


class XGBoostSurrogate(TrainableSurrogate, Surrogate):
    def __init__(self, data_model: DataModel, **kwargs) -> None:
        self.n_estimators = data_model.n_estimators
        self.max_depth = data_model.max_depth
        self.max_leaves = data_model.max_leaves
        self.max_bin = data_model.max_bin
        self.grow_policy = data_model.grow_policy
        self.learning_rate = data_model.learning_rate
        self.objective = data_model.objective
        self.booster = data_model.booster
        self.n_jobs = data_model.n_jobs
        self.gamma = data_model.gamma
        self.min_child_weight = data_model.min_child_weight
        self.max_delta_step = data_model.max_delta_step
        self.subsample = data_model.subsample
        self.sampling_method = data_model.sampling_method
        self.colsample_bytree = data_model.colsample_bytree
        self.colsample_bylevel = data_model.colsample_bylevel
        self.colsample_bynode = data_model.colsample_bynode
        self.reg_alpha = data_model.reg_alpha
        self.reg_lambda = data_model.reg_lambda
        self.scale_pos_weight = data_model.scale_pos_weight
        self.random_state = data_model.random_state
        self.num_parallel_tree = data_model.num_parallel_tree
        self.tmpfile_name = f"xgb_{uuid.uuid4().hex}.json"
        super().__init__(data_model=data_model, **kwargs)

    def _init_xgb(self):
        self.model = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            max_leaves=self.max_leaves,
            max_bin=self.max_bin,
            grow_policy=self.grow_policy,
            learning_rate=self.learning_rate,
            objective=self.objective,
            booster=self.booster,
            n_jobs=self.n_jobs,
            gamma=self.gamma,
            min_child_weight=self.min_child_weight,
            max_delta_step=self.max_delta_step,
            subsample=self.subsample,
            sampling_method=self.sampling_method,
            colsample_bytree=self.colsample_bytree,
            colsample_bylevel=self.colsample_bylevel,
            colsample_bynode=self.colsample_bynode,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            scale_pos_weight=self.scale_pos_weight,
            random_state=self.random_state,
            num_parallel_tree=self.num_parallel_tree,
        )

    def _fit(self, X: pd.DataFrame, Y: pd.DataFrame, **kwargs):
        transformed_X = self.inputs.transform(X, self.input_preprocessing_specs)
        self._init_xgb()
        self.model.fit(X=transformed_X.values, y=Y.values)

    def _predict(self, transformed_X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        preds = self.model.predict(transformed_X.values)
        return preds.reshape((transformed_X.shape[0], 1)), np.zeros(
            (transformed_X.shape[0], 1),
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
            with open(fname) as f:
                dump = f.read()
            return dump
