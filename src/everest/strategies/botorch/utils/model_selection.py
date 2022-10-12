import copy
from typing import Optional

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from everest.domain import Domain
from everest.domain.features import (CategoricalInputFeature,
                                     ContinuousOutputFeature, InputFeature)
from everest.strategies.botorch import tkwargs
from everest.strategies.botorch.utils.models import get_and_fit_model
from everest.strategies.strategy import KernelEnum, ScalerEnum
from hyperopt import fmin, hp, space_eval, tpe
from hyperopt.pyll import scope
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (make_scorer, mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import KFold, cross_validate, RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures


def cross_validate_gp(X,y,cv,metrics,gpkwargs):
    # fix this
    scores = {f"test_{m}":[] for m in metrics.keys()}
    for m in metrics.keys(): scores[f"train_{m}"]=[]
    
    for train_index, test_index in cv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        tX_train = torch.from_numpy(X_train).to(**tkwargs)
        tX_test = torch.from_numpy(X_test).to(**tkwargs)
        ty_train = torch.from_numpy(y_train).to(**tkwargs)

        # fit_gpytorch_model(mll)#, options = {"maxiter":1500})

        # learn model
        model = get_and_fit_model(
            train_X = tX_train,
            train_Y = ty_train.unsqueeze(-1),
            active_dims = list(range(X_train.shape[1])),
            cat_dims = [],
            scaler_name = ScalerEnum.NORMALIZE,
            use_categorical_kernel=False,
            cv = False,
            **gpkwargs
            #kernel_name = kernel_name,
            #use_ard = use_ard,
            #maxiter=1500
        )
        y_test_pred = model.posterior(X=tX_test).mean.cpu().detach().numpy().ravel()
        y_train_pred = model.posterior(X=tX_train).mean.cpu().detach().numpy().ravel()
        for scorer_name, scorer in metrics.items():
            scores[f"test_{scorer_name}"].append(scorer(y_test, y_test_pred))
            scores[f"train_{scorer_name}"].append(scorer(y_train, y_train_pred))
    return {key:np.array(value) for key, value in scores.items()}


class ModelSelector:

#TODO: RepeatedKFold move to repeated KFOLD and add plotting distributions

    def __init__(
        self, 
        domain: Domain, 
        experiments: pd.DataFrame, 
        cv_kwargs:dict= {"n_splits":5,  "n_repeats":1},#, random_state=5}
    ):
        self.domain = domain
        assert len(self.domain.get_features(CategoricalInputFeature)) == 0, "Categorical features not yet implemented."
        self.domain.validate_experiments(experiments)
        self.experiments = experiments
        self.cv_kwargs = cv_kwargs
        #
        self.metrics = {
            "r2" : r2_score,
            "mae": mean_absolute_error,
            "mape": mean_absolute_percentage_error,
            "mse": mean_squared_error,
        }
        return

    def _generate_performance_table(self):
        performance = {
                "feature": [],
                "model": [],
            }
        for m in self.metrics.keys(): 
            performance[f"test_{m}"] = []
            performance[f"train_{m}"] = []
        return performance

    def screen_sklearn(self, opt: bool = True) -> pd.DataFrame:
        performance = self._generate_performance_table()

        regressors = {
            "rf": RandomForestRegressor,
            "xgb": xgb.XGBRegressor,
        }

        spaces = {
            "rf": {
                'n_estimators': scope.int(hp.quniform('n_estimators', 2, 500,1)),
                'max_features': hp.choice("max_features", ["sqrt","log2"]),
                'max_depth':scope.int(hp.quniform('max_depth',2,100,2)),
                'min_samples_leaf':scope.int(hp.quniform('min_samples_leaf',1,5,1)),
                'min_samples_split':scope.int(hp.quniform('min_samples_split',2,10,1)),
                "bootstrap": hp.choice("bootstrap", [True, False]),
            },
            "xgb": {
                'eta': hp.uniform('eta', 0.01, 0.5),
                'max_depth': scope.int(hp.quniform("max_depth", 3, 18, 1)),
                'min_child_weight': hp.uniform('min_child_weight', 0.01, 0.75),
                'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
            }
        }  

        iterations = {
            "rf": 100,
            "xgb": 100,
        }
        
        def score(params):
            reg = Regressor(**params)
            scores = cross_validate(reg, X, y, cv=RepeatedKFold(**self.cv_kwargs),
                                    scoring={name: make_scorer(metric) for name, metric in self.metrics.items()},
                                    return_train_score=True)
            return scores["test_mae"].mean()

        for feat in self.domain.get_feature_keys(ContinuousOutputFeature):
            filtered_experiments = self.domain.preprocess_experiments_one_valid_output(self.experiments, feat)
            X = filtered_experiments[self.domain.get_feature_keys(InputFeature)].values
            y = filtered_experiments[feat].values
            for regressor_name, Regressor in regressors.items():
                # first build a cv model without any optimization
                scores = cross_validate(Regressor(), X, y, cv=RepeatedKFold(**self.cv_kwargs),
                                        scoring={name: make_scorer(metric) for name, metric in self.metrics.items()},
                                        return_train_score=True)
                performance["model"].append(regressor_name)
                performance["feature"].append(feat)
                for m in self.metrics.keys():
                    performance[f"test_{m}"].append(scores[f"test_{m}"].mean())
                    performance[f"train_{m}"].append(scores[f"train_{m}"].mean())
                if opt:
                    # second run hyperopt
                    best = fmin(
                        fn=score,
                        space=spaces[regressor_name],
                        max_evals = iterations[regressor_name],
                        algo=tpe.suggest
                    )
                    scores = cross_validate(Regressor(**space_eval(spaces[regressor_name],best)), X, y, cv=RepeatedKFold(**self.cv_kwargs),
                                            scoring={name: make_scorer(metric) for name, metric in self.metrics.items()},
                                            return_train_score=True)
                    performance["model"].append(f"{regressor_name}_opt")
                    performance["feature"].append(feat)
                    for m in self.metrics.keys():
                        performance[f"test_{m}"].append(scores[f"test_{m}"].mean())
                        performance[f"train_{m}"].append(scores[f"train_{m}"].mean())
        return pd.DataFrame.from_dict(performance)

    def screen_gps(self, opt:bool = True, feature_keys: Optional[list] = None) -> pd.DataFrame:

        performance = self._generate_performance_table()

        def score(params):
            scores= cross_validate_gp(X,y,cv=RepeatedKFold(**self.cv_kwargs),metrics=self.metrics,gpkwargs=params)
            return scores["test_mae"].mean()

        models = {
            "matern_25_ard": (KernelEnum.MATERN_25, True),
            "matern_15_ard": (KernelEnum.MATERN_15, True),
            "matern_05_ard": (KernelEnum.MATERN_05, True),
            "rbf_ard": (KernelEnum.RBF, True),
            "matern_25": (KernelEnum.MATERN_25, False),
            "matern_15": (KernelEnum.MATERN_15, False),
            "matern_05": (KernelEnum.MATERN_05, False),
            "rbf": (KernelEnum.RBF, False),
        }

        if feature_keys is None:
            #output_features = self.domain.get_feature_keys(ContinuousOutputFeature)
            feature_keys = self.domain.get_feature_keys(ContinuousOutputFeature)
        else:
            for key in feature_keys:
                feat = self.domain.get_feature(key)
                assert isinstance(feat, ContinuousOutputFeature), f"Feature {key} is not a ContinuousOutputFeature."


        #for feat in self.domain.get_feature_keys(ContinuousOutputFeature):
        for feat in feature_keys:           
            filtered_experiments = self.domain.preprocess_experiments_one_valid_output(self.experiments, feat)
            X = filtered_experiments[self.domain.get_feature_keys(InputFeature)].values
            y = filtered_experiments[feat].values

            for name, model_specs in models.items():
                gpkwargs = {
                    "kernel_name":model_specs[0],
                    "use_ard": model_specs[1]
                }
                # evaluate with default hyperparams
                scores= cross_validate_gp(X,y,cv=RepeatedKFold(**self.cv_kwargs),metrics=self.metrics,gpkwargs=gpkwargs)
                for m in self.metrics.keys():
                    performance[f"test_{m}"].append(scores[f"test_{m}"].mean())
                    performance[f"train_{m}"].append(scores[f"train_{m}"].mean())
                performance["model"].append(f"{name}")
                performance["feature"].append(feat)

                if opt:
                    space = {"maxiter": scope.int(hp.quniform('maxiter', 5, 15000, 1)),}
                    space.update(gpkwargs)

                    best = fmin(
                        fn=score,
                        space = space,
                        max_evals = 10,
                        algo=tpe.suggest,
                    )
                    scores= cross_validate_gp(X,y,cv=RepeatedKFold(**self.cv_kwargs),metrics=self.metrics,gpkwargs=space_eval(space,best))
                    for m in self.metrics.keys():
                        performance[f"test_{m}"].append(scores[f"test_{m}"].mean())
                        performance[f"train_{m}"].append(scores[f"train_{m}"].mean())
                    performance["model"].append(f"{name}_opt")
                    performance["feature"].append(feat)

            
            # linear models as baselines
            ## no interaction
            reg = make_pipeline(MinMaxScaler(),LinearRegression())
            scores = cross_validate(reg, X, y, cv=RepeatedKFold(**self.cv_kwargs),
                                    scoring={name: make_scorer(metric)for name, metric in self.metrics.items()},
                                    return_train_score=True)
            performance["model"].append("linear")
            performance["feature"].append(feat)
            for m in self.metrics.keys():
                performance[f"test_{m}"].append(scores[f"test_{m}"].mean())
                performance[f"train_{m}"].append(scores[f"train_{m}"].mean())
            ## all interactions
            reg = make_pipeline(MinMaxScaler(), PolynomialFeatures(degree=2, interaction_only=True), LinearRegression())
            scores = cross_validate(reg, X, y, cv=RepeatedKFold(**self.cv_kwargs),
                                    scoring={name: make_scorer(metric)for name, metric in self.metrics.items()},
                                    return_train_score=True)
            performance["model"].append("linear_interaction")
            performance["feature"].append(feat)
            for m in self.metrics.keys():
                performance[f"test_{m}"].append(scores[f"test_{m}"].mean())
                performance[f"train_{m}"].append(scores[f"train_{m}"].mean())           
        return pd.DataFrame.from_dict(performance)
