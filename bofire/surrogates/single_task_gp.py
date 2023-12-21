from typing import Dict, Optional, Union

import botorch
import pandas as pd
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.input import InputStandardize, Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

import bofire.kernels.api as kernels
import bofire.priors.api as priors
from bofire.data_models.domain.api import Inputs
from bofire.data_models.enum import CategoricalEncodingEnum, OutputFilteringEnum
from bofire.data_models.features.api import TInputTransformSpecs
from bofire.data_models.molfeatures.api import MolFeatures
from bofire.data_models.surrogates.api import SingleTaskGPSurrogate as DataModel
from bofire.data_models.surrogates.scaler import ScalerEnum
from bofire.surrogates.botorch import BotorchSurrogate
from bofire.surrogates.trainable import TrainableSurrogate
from bofire.utils.torch_tools import tkwargs


def get_scaler(
    inputs: Inputs,
    input_preprocessing_specs: TInputTransformSpecs,
    scaler: ScalerEnum,
    X: pd.DataFrame,
) -> Union[InputStandardize, Normalize, None]:
    """Returns the instanitated scaler object for a set of input features and
    input_preprocessing_specs.


    Args:
        inputs (Inputs): Input features.
        input_preprocessing_specs (TInputTransformSpecs): Dictionary how to treat
            the categoricals and/or molecules.
        scaler (ScalerEnum): Enum indicating the scaler of interest.
        X (pd.DataFrame): The dataset of interest.

    Returns:
        Union[InputStandardize, Normalize]: The instantiated scaler class
    """
    if scaler != ScalerEnum.IDENTITY:
        features2idx, _ = inputs._get_transform_info(input_preprocessing_specs)

        d = 0
        for indices in features2idx.values():
            d += len(indices)

        non_numerical_features = [
            key
            for key, value in input_preprocessing_specs.items()
            if value != CategoricalEncodingEnum.DESCRIPTOR
            and not isinstance(value, MolFeatures)
        ]

        ord_dims = []
        for feat in inputs.get():
            if feat.key not in non_numerical_features:
                ord_dims += features2idx[feat.key]

        if scaler == ScalerEnum.NORMALIZE:
            lower, upper = inputs.get_bounds(
                specs=input_preprocessing_specs, experiments=X
            )
            scaler_transform = Normalize(
                d=d,
                bounds=torch.tensor([lower, upper]).to(**tkwargs),
                indices=ord_dims,
                batch_shape=torch.Size(),
            )
        elif scaler == ScalerEnum.STANDARDIZE:
            scaler_transform = InputStandardize(
                d=d,
                indices=ord_dims,
                batch_shape=torch.Size(),
            )
        else:
            raise ValueError("Scaler enum not known.")
        return scaler_transform
    else:
        return None


class SingleTaskGPSurrogate(BotorchSurrogate, TrainableSurrogate):
    def __init__(
        self,
        data_model: DataModel,
        **kwargs,
    ):
        self.kernel = data_model.kernel
        self.scaler = data_model.scaler
        self.output_scaler = data_model.output_scaler
        self.noise_prior = data_model.noise_prior
        super().__init__(data_model=data_model, **kwargs)

    model: Optional[botorch.models.SingleTaskGP] = None
    _output_filtering: OutputFilteringEnum = OutputFilteringEnum.ALL
    training_specs: Dict = {}

    def _fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        scaler = get_scaler(self.inputs, self.input_preprocessing_specs, self.scaler, X)
        transformed_X = self.inputs.transform(X, self.input_preprocessing_specs)

        tX, tY = torch.from_numpy(transformed_X.values).to(**tkwargs), torch.from_numpy(
            Y.values
        ).to(**tkwargs)

        self.model = botorch.models.SingleTaskGP(  # type: ignore
            train_X=tX,
            train_Y=tY,
            covar_module=kernels.map(
                self.kernel,
                batch_shape=torch.Size(),
                active_dims=list(range(tX.shape[1])),
                ard_num_dims=1,  # this keyword is ingored
            ),
            outcome_transform=Standardize(m=tY.shape[-1])
            if self.output_scaler == ScalerEnum.STANDARDIZE
            else None,
            input_transform=scaler,
        )

        self.model.likelihood.noise_covar.noise_prior = priors.map(self.noise_prior)  # type: ignore

        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll, options=self.training_specs, max_attempts=10)
