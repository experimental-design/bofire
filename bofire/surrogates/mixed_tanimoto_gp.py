import base64
import io
import warnings
from functools import partial
from typing import Callable, Dict, List, Optional

import pandas as pd
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.kernels.categorical import CategoricalKernel
from botorch.models.transforms.input import (
    ChainedInputTransform,
    InputTransform,
    OneHotToNumeric,
)
from botorch.models.transforms.outcome import OutcomeTransform, Standardize
from botorch.utils.transforms import normalize_indices
from gpytorch.constraints import GreaterThan
from gpytorch.kernels.kernel import Kernel
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior
from torch import Tensor

import bofire.kernels.api as kernels
from bofire.data_models.enum import (
    CategoricalEncodingEnum,
    OutputFilteringEnum,
)
from bofire.data_models.molfeatures.api import (
    Fingerprints,
    FingerprintsFragments,
    Fragments,
    MordredDescriptors,
)

# MolecularEncodingEnum,
from bofire.data_models.surrogates.api import MixedTanimotoGPSurrogate as DataModel

# from bofire.data_models.kernels.categorical import HammondDistanceKernel
from bofire.surrogates.botorch import BotorchSurrogate
from bofire.surrogates.single_task_gp import get_scaler
from bofire.surrogates.trainable import TrainableSurrogate
from bofire.utils.torch_tools import tkwargs


class MixedTanimotoGP(SingleTaskGP):
    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        mol_dims: List[int],
        mol_kernel_factory: Callable[[torch.Size, int, List[int]], Kernel],
        cat_dims: Optional[List[int]] = None,
        # cat_kernel_factory: Optional[
        #    Callable[[torch.Size, int, List[int]], Kernel]
        # ] = None, --> BoTorch forced to use CategoricalKernel
        cont_kernel_factory: Optional[
            Callable[[torch.Size, int, List[int]], Kernel]
        ] = None,
        likelihood: Optional[Likelihood] = None,
        outcome_transform: Optional[OutcomeTransform] = None,  # TODO
        input_transform: Optional[InputTransform] = None,  # TODO
    ) -> None:
        if len(mol_dims) == 0:
            raise ValueError("Must specify molecular dimensions for MixedTanimotoGP")

        if cat_dims is None:
            cat_dims = []
        self._ignore_X_dims_scaling_check = cat_dims

        _, aug_batch_shape = self.get_batch_dimensions(train_X=train_X, train_Y=train_Y)

        d = train_X.shape[-1]
        mol_dims = normalize_indices(indices=mol_dims, d=d)
        cat_dims = normalize_indices(indices=cat_dims, d=d)
        ord_dims = sorted(set(range(d)) - set(cat_dims) - set(mol_dims))

        if cont_kernel_factory is None:
            cont_kernel_factory = kernels.map_MaternKernel(
                data_model=MaternKernel(ard=True, nu=2.5),
                batch_shape=aug_batch_shape,
                ard_num_dims=len(ord_dims),
                active_dims=ord_dims,
            )

        if likelihood is None:
            min_noise = 1e-5 if train_X.dtype == torch.float else 1e-6
            likelihood = GaussianLikelihood(
                batch_shape=aug_batch_shape,
                noise_constraint=GreaterThan(
                    min_noise, transform=None, initial_value=1e-3
                ),
                noise_prior=GammaPrior(0.9, 10.0),
            )

        if len(ord_dims) == 0:
            sum_kernel = ScaleKernel(
                CategoricalKernel(
                    batch_shape=aug_batch_shape,
                    ard_num_dims=len(cat_dims),
                    active_dims=cat_dims,
                    lengthscale_constraint=GreaterThan(1e-06),
                )
            ) + ScaleKernel(
                mol_kernel_factory(
                    batch_shape=aug_batch_shape,
                    ard_num_dims=len(mol_dims),
                    active_dims=mol_dims,
                )
            )

            prod_kernel = ScaleKernel(
                CategoricalKernel(
                    batch_shape=aug_batch_shape,
                    ard_num_dims=len(cat_dims),
                    active_dims=cat_dims,
                    lengthscale_constraint=GreaterThan(1e-06),
                )
            ) * ScaleKernel(
                mol_kernel_factory(
                    batch_shape=aug_batch_shape,
                    ard_num_dims=len(mol_dims),
                    active_dims=mol_dims,
                )
            )

            covar_module = sum_kernel + prod_kernel

        elif len(cat_dims) == 0:
            sum_kernel = ScaleKernel(
                cont_kernel_factory(
                    batch_shape=aug_batch_shape,
                    ard_num_dims=len(ord_dims),
                    active_dims=ord_dims,
                )
            ) + ScaleKernel(
                mol_kernel_factory(
                    batch_shape=aug_batch_shape,
                    ard_num_dims=len(mol_dims),
                    active_dims=mol_dims,
                )
            )

            prod_kernel = ScaleKernel(
                cont_kernel_factory(
                    batch_shape=aug_batch_shape,
                    ard_num_dims=len(ord_dims),
                    active_dims=ord_dims,
                )
            ) * ScaleKernel(
                mol_kernel_factory(
                    batch_shape=aug_batch_shape,
                    ard_num_dims=len(mol_dims),
                    active_dims=mol_dims,
                )
            )

            covar_module = sum_kernel + prod_kernel

        else:
            sum_kernel = (
                ScaleKernel(
                    cont_kernel_factory(
                        batch_shape=aug_batch_shape,
                        ard_num_dims=len(ord_dims),
                        active_dims=ord_dims,
                    )
                )
                + ScaleKernel(
                    mol_kernel_factory(
                        batch_shape=aug_batch_shape,
                        ard_num_dims=len(mol_dims),
                        active_dims=mol_dims,
                    )
                )
                + ScaleKernel(
                    CategoricalKernel(
                        batch_shape=aug_batch_shape,
                        ard_num_dims=len(cat_dims),
                        active_dims=cat_dims,
                        lengthscale_constraint=GreaterThan(1e-06),
                    )
                )
            )

            prod_kernel = (
                ScaleKernel(
                    cont_kernel_factory(
                        batch_shape=aug_batch_shape,
                        ard_num_dims=len(ord_dims),
                        active_dims=ord_dims,
                    )
                )
                * ScaleKernel(
                    mol_kernel_factory(
                        batch_shape=aug_batch_shape,
                        ard_num_dims=len(mol_dims),
                        active_dims=mol_dims,
                    )
                )
                * ScaleKernel(
                    CategoricalKernel(
                        batch_shape=aug_batch_shape,
                        ard_num_dims=len(cat_dims),
                        active_dims=cat_dims,
                        lengthscale_constraint=GreaterThan(1e-06),
                    )
                )
            )
            covar_module = sum_kernel + prod_kernel

        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            likelihood=likelihood,
            covar_module=covar_module,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )


class MixedTanimotoGPSurrogate(BotorchSurrogate, TrainableSurrogate):
    def __init__(
        self,
        data_model: DataModel,
        **kwargs,
    ):
        self.continuous_kernel = data_model.continuous_kernel
        self.categorical_kernel = data_model.categorical_kernel
        self.molecular_kernel = data_model.molecular_kernel
        self.scaler = data_model.scaler
        super().__init__(data_model=data_model, **kwargs)

    model: Optional[MixedTanimotoGP] = None
    _output_filtering: OutputFilteringEnum = OutputFilteringEnum.ALL
    training_specs: Dict = {}

    def _fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        # Categorical inputs that are not descriptor-based
        categorical_features_list = [
            key
            for key, value in self.input_preprocessing_specs.items()
            if value != CategoricalEncodingEnum.DESCRIPTOR
            and not isinstance(value, Fingerprints)
            and not isinstance(value, Fragments)
            and not isinstance(value, FingerprintsFragments)
            and not isinstance(value, MordredDescriptors)
        ]

        # Molecular features only include fingerprints, fragments, and fingerprintsfragments
        molecular_features_list = [
            key
            for key, value in self.input_preprocessing_specs.items()
            if isinstance(value, Fingerprints)
            or isinstance(value, Fragments)
            or isinstance(value, FingerprintsFragments)
        ]

        # Continuous features include continuous inputs, categorical inputs with descriptors, and Mordred descriptors
        continuous_features_list = [
            feat.key
            for feat in self.inputs.get()
            if feat.key not in categorical_features_list
            and feat.key not in molecular_features_list
        ]

        if len(continuous_features_list) == 0:
            scaler = None  # skip the scaler
        else:
            scaler = get_scaler(
                self.inputs, self.input_preprocessing_specs, self.scaler, X
            )

        transformed_X = self.inputs.transform(X, self.input_preprocessing_specs)

        tX, tY = torch.from_numpy(transformed_X.values).to(**tkwargs), torch.from_numpy(
            Y.values
        ).to(**tkwargs)

        features2idx, _ = self.inputs._get_transform_info(
            self.input_preprocessing_specs
        )

        # List of indexes for Molecular features using Fingerprints, Fragment, FingerprintsFragments
        mol_dims = []
        for mol_feat in molecular_features_list:
            for i in features2idx[mol_feat]:
                mol_dims.append(i)

        # List of indexes for Continuous inputs, Categorical inputs with descriptors, Mordred descriptors
        ord_dims = []
        for ord_feat in continuous_features_list:
            for i in features2idx[ord_feat]:
                ord_dims.append(i)

        # these are the categorical dimensions after applying the OneHotToNumeric transform
        cat_dims = list(
            range(
                len(ord_dims) + len(mol_dims),
                len(ord_dims) + len(mol_dims) + len(categorical_features_list),
            )
        )

        if len(categorical_features_list) == 0:
            tf = scaler
            tXX = tX
        else:
            # these are the categorical features within the OneHotToNumeric transform
            categorical_features = {
                features2idx[feat][0]: len(features2idx[feat])
                for feat in categorical_features_list
            }

            o2n = OneHotToNumeric(
                dim=tX.shape[1],
                categorical_features=categorical_features,
                transform_on_train=False,
            )
            tf = (
                ChainedInputTransform(tf1=scaler, tf2=o2n)
                if scaler is not None
                else o2n
            )
            tXX = o2n.transform(tX)

        # fit the model
        self.model = MixedTanimotoGP(
            train_X=tXX,
            train_Y=tY,
            cat_dims=cat_dims,
            mol_dims=mol_dims,
            cont_kernel_factory=partial(kernels.map, data_model=self.continuous_kernel),
            # cat_kernel_factory=partial(kernels.map, data_model=self.categorical_kernel), BoTorch forced to use CategoricalKernel
            mol_kernel_factory=partial(kernels.map, data_model=self.molecular_kernel),
            outcome_transform=Standardize(m=tY.shape[-1]),
            input_transform=tf,
        )
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll, options=self.training_specs, max_attempts=10)

    def _dumps(self) -> str:
        """Dumps the actual model to a string via pickle as this is not directly json serializable."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import bofire.surrogates.cloudpickle_module as cloudpickle_module

            if len(w) == 1:
                raise ModuleNotFoundError("Cloudpickle is not available.")

        buffer = io.BytesIO()
        torch.save(self.model, buffer, pickle_module=cloudpickle_module)  # type: ignore
        return base64.b64encode(buffer.getvalue()).decode()

    def loads(self, data: str):
        """Loads the actual model from a base64 encoded pickle bytes object and writes it to the `model` attribute."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import bofire.surrogates.cloudpickle_module as cloudpickle_module

            if len(w) == 1:
                raise ModuleNotFoundError("Cloudpickle is not available.")

        buffer = io.BytesIO(base64.b64decode(data.encode()))
        self.model = torch.load(buffer, pickle_module=cloudpickle_module)  # type: ignore
