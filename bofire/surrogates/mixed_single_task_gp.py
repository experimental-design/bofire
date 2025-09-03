from functools import partial
from typing import Dict, Optional

import botorch
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from gpytorch.mlls import ExactMarginalLogLikelihood

import bofire.kernels.api as kernels
import bofire.priors.api as priors
from bofire.data_models.enum import CategoricalEncodingEnum, OutputFilteringEnum
from bofire.data_models.features.api import CategoricalInput
from bofire.data_models.surrogates.api import MixedSingleTaskGPSurrogate as DataModel
from bofire.surrogates.botorch import TrainableBotorchSurrogate


class MixedSingleTaskGPSurrogate(TrainableBotorchSurrogate):
    def __init__(
        self,
        data_model: DataModel,
        **kwargs,
    ):
        self.continuous_kernel = data_model.continuous_kernel
        self.categorical_kernel = data_model.categorical_kernel
        self.noise_prior = data_model.noise_prior
        self.scaler = data_model.scaler
        self.output_scaler = data_model.output_scaler
        super().__init__(data_model=data_model, **kwargs)

    model: Optional[botorch.models.MixedSingleTaskGP] = None
    _output_filtering: OutputFilteringEnum = OutputFilteringEnum.ALL
    # features2idx: Optional[Dict] = None
    # non_numerical_features: Optional[List] = None
    training_specs: Dict = {}

    def _fit_botorch(
        self,
        tX: torch.Tensor,
        tY: torch.Tensor,
        input_transform: Optional[InputTransform] = None,
        outcome_transform: Optional[OutcomeTransform] = None,
        **kwargs,
    ):
        # get categorical features without an encoding in categorical_encodings
        categorical_feature_keys = [
            key
            for key in self.inputs.get_keys(CategoricalInput)
            if self.categorical_encodings.get(key, CategoricalEncodingEnum.ORDINAL)
            == CategoricalEncodingEnum.ORDINAL
        ]
        assert len(categorical_feature_keys) > 0
        # get indices for categorical feature keys
        categorical_feature_indices = self.inputs.get_feature_indices(
            self.categorical_encodings, categorical_feature_keys
        )
        print(categorical_feature_indices)

        # fit the model
        self.model = botorch.models.MixedSingleTaskGP(
            train_X=tX,
            train_Y=tY,
            cat_dims=categorical_feature_indices,
            # cont_kernel_factory=self.continuous_kernel.to_gpytorch,
            cont_kernel_factory=partial(
                kernels.map,
                data_model=self.continuous_kernel,
                features_to_idx_mapper=lambda feats: self.inputs.get_feature_indices(
                    self.categorical_encodings, feats
                ),
            ),
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )
        self.model.likelihood.noise_covar.noise_prior = priors.map(self.noise_prior)

        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll, options=self.training_specs)
