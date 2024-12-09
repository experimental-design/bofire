import warnings
from typing import Dict, Optional

import botorch
import numpy as np
import pandas as pd
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

import bofire.kernels.api as kernels
import bofire.priors.api as priors
from bofire.data_models.enum import OutputFilteringEnum
from bofire.data_models.features.api import TaskInput
from bofire.data_models.priors.api import LKJPrior

# from bofire.data_models.molfeatures.api import MolFeatures
from bofire.data_models.surrogates.api import MultiTaskGPSurrogate as DataModel
from bofire.data_models.surrogates.scaler import ScalerEnum
from bofire.surrogates.botorch import BotorchSurrogate
from bofire.surrogates.trainable import TrainableSurrogate
from bofire.surrogates.utils import get_scaler
from bofire.utils.torch_tools import tkwargs


class MultiTaskGPSurrogate(BotorchSurrogate, TrainableSurrogate):
    def __init__(
        self,
        data_model: DataModel,
        **kwargs,
    ):
        self.n_tasks = len(data_model.inputs.get(TaskInput).features[0].categories)
        self.kernel = data_model.kernel
        self.scaler = data_model.scaler
        self.output_scaler = data_model.output_scaler
        self.noise_prior = data_model.noise_prior
        self.task_prior = data_model.task_prior
        if isinstance(self.task_prior, LKJPrior):
            # set the number of tasks in the prior
            self.task_prior.n_tasks = self.n_tasks
        # obtain the name of the task feature
        self.task_feature_key = data_model.inputs.get_keys(TaskInput)[0]

        super().__init__(data_model=data_model, **kwargs)

    model: Optional[botorch.models.MultiTaskGP] = None
    _output_filtering: OutputFilteringEnum = OutputFilteringEnum.ALL
    training_specs: Dict = {}

    def _fit(self, X: pd.DataFrame, Y: pd.DataFrame):  # type: ignore
        scaler = get_scaler(self.inputs, self.input_preprocessing_specs, self.scaler, X)
        transformed_X = self.inputs.transform(X, self.input_preprocessing_specs)

        tX, tY = (
            torch.from_numpy(transformed_X.values).to(**tkwargs),
            torch.from_numpy(Y.values).to(**tkwargs),
        )

        self.model = botorch.models.MultiTaskGP(
            train_X=tX,
            train_Y=tY,
            task_feature=transformed_X.columns.get_loc(  # type: ignore
                self.task_feature_key,
            ),  # obtain the fidelity index
            covar_module=kernels.map(
                self.kernel,
                batch_shape=torch.Size(),
                active_dims=list(
                    range(tX.shape[1] - 1),
                ),  # kernel is for input space so we subtract one for the fidelity index
                ard_num_dims=1,  # this keyword is ignored
                features_to_idx_mapper=lambda feats: self.inputs.get_feature_indices(
                    self.input_preprocessing_specs, feats
                ),
            ),
            outcome_transform=(
                Standardize(m=tY.shape[-1])
                if self.output_scaler == ScalerEnum.STANDARDIZE
                else None
            ),
            input_transform=scaler,
        )

        if isinstance(self.task_prior, LKJPrior):
            warnings.warn(
                "The LKJ prior has issues when sampling from the prior, prior has been defaulted to None.",
                UserWarning,
            )
            # once the issue is fixed, the following line should be uncommented
            # self.model.task_covar_module.register_prior(
            #     "IndexKernelPrior", priors.map(self.lkj_prior), _index_kernel_prior_closure
            # )
        self.model.likelihood.noise_covar.noise_prior = priors.map(self.noise_prior)

        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll, options=self.training_specs, max_attempts=10)

    def _predict(self, transformed_X: pd.DataFrame):
        # transform to tensor
        X = torch.from_numpy(transformed_X.values).to(**tkwargs)
        with torch.no_grad():
            try:
                posterior = self.model.posterior(X=X, observation_noise=True)  # type: ignore
            except NotImplementedError:
                posterior = self.model.posterior(X=X, observation_noise=False)  # type: ignore
            preds = posterior.mean.cpu().detach().numpy()
            stds = np.sqrt(posterior.variance.cpu().detach().numpy())

        return preds, stds


def _index_kernel_prior_closure(m):
    return m._eval_covar_matrix()
