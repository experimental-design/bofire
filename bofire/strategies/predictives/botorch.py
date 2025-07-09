from abc import abstractmethod
from typing import Callable, Dict, List, Optional, Tuple, get_args

import numpy as np
import pandas as pd
import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.utils import get_infeasible_cost
from botorch.models.gpytorch import GPyTorchModel
from torch import Tensor

from bofire.data_models.features.api import Input
from bofire.data_models.strategies.api import BotorchStrategy as DataModel
from bofire.data_models.strategies.api import RandomStrategy as RandomStrategyDataModel
from bofire.data_models.surrogates.api import AnyTrainableSurrogate
from bofire.data_models.types import InputTransformSpecs
from bofire.outlier_detection.outlier_detections import OutlierDetections
from bofire.strategies.predictives.acqf_optimization import (
    AcquisitionOptimizer,
    get_optimizer,
)
from bofire.strategies.predictives.predictive import PredictiveStrategy
from bofire.strategies.random import RandomStrategy
from bofire.surrogates.botorch_surrogates import BotorchSurrogates
from bofire.utils.torch_tools import tkwargs


class BotorchStrategy(PredictiveStrategy):
    def __init__(
        self,
        data_model: DataModel,
        **kwargs,
    ):
        super().__init__(data_model=data_model, **kwargs)

        self.acqf_optimizer: AcquisitionOptimizer = get_optimizer(
            data_model.acquisition_optimizer
        )

        self.surrogate_specs = data_model.surrogate_specs
        if data_model.outlier_detection_specs is not None:
            self.outlier_detection_specs = OutlierDetections(
                data_model=data_model.outlier_detection_specs,
            )
        else:
            self.outlier_detection_specs = None
        self.min_experiments_before_outlier_check = (
            data_model.min_experiments_before_outlier_check
        )
        self.frequency_check = data_model.frequency_check
        self.frequency_hyperopt = data_model.frequency_hyperopt
        self.folds = data_model.folds
        self.surrogates = None

        torch.manual_seed(self.seed)

    model: Optional[GPyTorchModel] = None

    @property
    def input_preprocessing_specs(self) -> InputTransformSpecs:
        return self.surrogate_specs.input_preprocessing_specs

    @property
    def _features2names(self) -> Dict[str, Tuple[str]]:
        _, features2names = self.domain.inputs._get_transform_info(
            self.input_preprocessing_specs,
        )
        return features2names

    def _fit(self, experiments: pd.DataFrame):
        """[summary]

        Args:
            transformed (pd.DataFrame): [description]

        """
        # perform outlier detection
        if self.outlier_detection_specs is not None:
            if (
                self.num_experiments >= self.min_experiments_before_outlier_check
                and self.num_experiments % self.frequency_check == 0
            ):
                experiments = self.outlier_detection_specs.detect(experiments)
        # perform hyperopt
        if (self.frequency_hyperopt > 0) and (
            self.num_experiments % self.frequency_hyperopt == 0
        ):
            # we have to import here to avoid circular imports
            from bofire.runners.hyperoptimize import hyperoptimize

            self.surrogate_specs.surrogates = [  # type: ignore
                (
                    hyperoptimize(
                        surrogate_data=surrogate_data,  # type: ignore
                        training_data=experiments,
                        folds=self.folds,
                    )[0]
                    if isinstance(surrogate_data, get_args(AnyTrainableSurrogate))
                    else surrogate_data
                )
                for surrogate_data in self.surrogate_specs.surrogates
            ]

        # map the surrogate spec, we keep it here as attribute to be able to save/dump
        # the surrogate
        self.surrogates = BotorchSurrogates(data_model=self.surrogate_specs)

        self.surrogates.fit(experiments)
        self.model = self.surrogates.compatibilize(  # type: ignore
            inputs=self.domain.inputs,
            outputs=self.domain.outputs,
        )

    def _predict(self, transformed: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:  # type: ignore
        # we are using self.model here for this purpose we have to take the transformed
        # input and further transform it to a torch tensor
        X = torch.from_numpy(transformed.values).to(**tkwargs)
        with torch.no_grad():
            try:
                posterior = self.model.posterior(X=X, observation_noise=True)  # type: ignore
            except (
                NotImplementedError
            ):  # NotImplementedEerror is thrown for MultiTaskGPSurrogate
                posterior = self.model.posterior(X=X, observation_noise=False)  # type: ignore

            if len(posterior.mean.shape) == 2:
                preds = posterior.mean.cpu().detach().numpy()
                stds = np.sqrt(posterior.variance.cpu().detach().numpy())
            elif len(posterior.mean.shape) == 3:
                preds = posterior.mean.mean(dim=0).cpu().detach().numpy()
                stds = np.sqrt(posterior.variance.mean(dim=0).cpu().detach().numpy())
            else:
                raise ValueError("Wrong dimension of posterior mean. Expecting 2 or 3.")
        return preds, stds

    def calc_acquisition(
        self,
        candidates: pd.DataFrame,
        combined: bool = False,
    ) -> np.ndarray:
        """Calculate the acquisition value for a set of experiments.

        Args:
            candidates (pd.DataFrame): Dataframe with experimentes for which the acqf value should be calculated.
            combined (bool, optional): If combined an acquisition value for the whole batch is calculated, else individual ones.
                Defaults to False.

        Returns:
            np.ndarray: Dataframe with the acquisition values.

        """
        acqf = self._get_acqfs(1)[0]

        transformed = self.domain.inputs.transform(
            candidates,
            self.input_preprocessing_specs,
        )
        X = torch.from_numpy(transformed.values).to(**tkwargs)
        if combined is False:
            X = X.unsqueeze(-2)

        with torch.no_grad():
            vals = acqf.forward(X).cpu().detach().numpy()

        return vals

    def _ask(self, candidate_count: int) -> pd.DataFrame:  # type: ignore
        """[summary]

        Args:
            candidate_count (int, optional): [description]. Defaults to 1.

        Returns:
            pd.DataFrame: [description]

        """
        assert candidate_count > 0, "candidate_count has to be larger than zero."
        if self.experiments is None:
            raise ValueError("No experiments have been provided yet.")

        acqfs = self._get_acqfs(candidate_count)

        candidates = self.acqf_optimizer.optimize(
            candidate_count,
            acqfs,
            self.domain,
            self.input_preprocessing_specs,
            self.experiments,
        )

        return candidates

    def _tell(self) -> None:
        pass

    @abstractmethod
    def _get_acqfs(self, n: int) -> List[AcquisitionFunction]:
        pass

    def has_sufficient_experiments(
        self,
    ) -> bool:
        if self.experiments is None:
            return False
        if (
            len(
                self.domain.outputs.preprocess_experiments_all_valid_outputs(
                    experiments=self.experiments,
                ),
            )
            > 1
        ):
            return True
        return False

    def get_acqf_input_tensors(self):
        assert self.experiments is not None
        experiments = self.domain.outputs.preprocess_experiments_all_valid_outputs(
            self.experiments,
        )

        clean_experiments = experiments.drop_duplicates(
            subset=[var.key for var in self.domain.inputs.get(Input)],
            keep="first",
            inplace=False,
        )
        # we should only provide those experiments to the acqf builder in which all
        # input constraints are fulfilled, output constraints are handled directly
        # in botorch
        clean_experiments = clean_experiments[
            self.domain.is_fulfilled(clean_experiments)
        ].copy()
        if len(clean_experiments) == 0:
            raise ValueError(
                "No valid and feasible experiments are available for setting up the acquisition function. Check your constraints.",
            )

        transformed = self.domain.inputs.transform(
            clean_experiments,
            self.input_preprocessing_specs,
        )
        X_train = torch.from_numpy(transformed.values).to(**tkwargs)

        if self.candidates is not None:
            transformed_candidates = self.domain.inputs.transform(
                self.candidates,
                self.input_preprocessing_specs,
            )
            X_pending = torch.from_numpy(transformed_candidates.values).to(**tkwargs)
        else:
            X_pending = None

        return X_train, X_pending

    def get_infeasible_cost(
        self,
        objective: Callable[[Tensor, Tensor], Tensor],
        n_samples=128,
    ) -> Tensor:
        X_train, X_pending = self.get_acqf_input_tensors()
        sampler = RandomStrategy(data_model=RandomStrategyDataModel(domain=self.domain))
        samples = sampler.ask(candidate_count=n_samples)
        # we need to transform the samples
        transformed_samples = torch.from_numpy(
            self.domain.inputs.transform(
                samples, self.input_preprocessing_specs
            ).values,
        ).to(**tkwargs)
        X = (
            torch.cat((X_train, X_pending, transformed_samples))
            if X_pending is not None
            else torch.cat((X_train, transformed_samples))
        )
        assert self.model is not None
        return get_infeasible_cost(
            X=X,
            model=self.model,
            objective=objective,  # type: ignore
        )
