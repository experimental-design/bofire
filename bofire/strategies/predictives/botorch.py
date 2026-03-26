import warnings
from abc import abstractmethod
from typing import Callable, Dict, List, Optional, Tuple, get_args

import numpy as np
import pandas as pd
import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.utils import get_infeasible_cost
from botorch.models.gpytorch import GPyTorchModel
from torch import Tensor

from bofire.data_models.constraints.api import NChooseKConstraint
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
        self.include_infeasible_exps_in_acqf_calc = (
            data_model.include_infeasible_exps_in_acqf_calc
        )

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

            self.surrogate_specs.surrogates = [  # ty: ignore[invalid-assignment]
                (
                    hyperoptimize(
                        surrogate_data=surrogate_data,
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
        re_init_kwargs = (
            self.surrogates.re_init_kwargs if self.surrogates is not None else None
        )
        self.surrogates = BotorchSurrogates(
            data_model=self.surrogate_specs, re_init_kwargs=re_init_kwargs
        )

        self.surrogates.fit(experiments)
        self.model = self.surrogates.compatibilize(
            inputs=self.domain.inputs,
            outputs=self.domain.outputs,
        )

    def _predict(self, transformed: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        # we are using self.model here for this purpose we have to take the transformed
        # input and further transform it to a torch tensor
        X = torch.from_numpy(transformed.values).to(**tkwargs)
        with torch.no_grad():
            try:
                posterior = self.model.posterior(X=X, observation_noise=True)
            except (
                NotImplementedError
            ):  # NotImplementedEerror is thrown for MultiTaskGPSurrogate
                posterior = self.model.posterior(X=X, observation_noise=False)

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

    def _ask(self, candidate_count: int) -> pd.DataFrame:
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
        self._current_acqfs = acqfs

        candidates = self.acqf_optimizer.optimize(
            candidate_count,
            acqfs,
            self.domain,
            self.experiments,
        )

        return candidates

    def _tell(self) -> None:
        pass

    def _postprocess_candidates(self, candidates: pd.DataFrame) -> pd.DataFrame:
        if self.domain.is_nchoosek_pruning_applicable():
            candidates = self._prune_nchoosek_candidates(candidates)
        return super()._postprocess_candidates(candidates)

    def _prune_nchoosek_candidates(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """Apply greedy pruning to satisfy NChooseK constraints.

        Based on the BONSAI algorithm (https://arxiv.org/abs/2602.07144).
        For each candidate in the batch, greedily zeros the feature with the
        smallest acquisition function impact until all NChooseK constraints are
        satisfied. Later candidates are conditioned on already-pruned ones.

        Currently this is only applicable when no NChooseK feature participates
        in any other constraint (see Domain.is_nchoosek_pruning_applicable).

        Future work: extend pruning to handle NChooseK features that also
        appear in linear equality/inequality constraints. The idea is to replace
        the simple "set x_k = 0" variant construction with:
        1. Zero feature k and QP-project the candidate onto the feasible set
           of all linear constraints with x_k = 0 fixed. This serves both as a
           feasibility check (skip k if infeasible) and as a warm-start.
        2. Run a local optimize_acqf call with fixed_features={k: 0}, using the
           QP-projected point as the initial condition, subject to all linear
           constraints. This finds the best achievable AF value with feature k
           zeroed, giving a much better signal for the greedy decision than
           projection alone.
        3. For q > 1, the local optimization should condition on already-pruned
           candidates to preserve the joint AF evaluation.
        The QP infrastructure for step 1 already exists in BoFire's
        LinearProjection repair function used by the GeneticAlgorithmOptimizer.

        Args:
            candidates: DataFrame with candidate proposals.

        Returns:
            DataFrame with pruned candidates satisfying NChooseK constraints.
        """
        acqf = self._current_acqfs[0]
        nchoosek_constraints = self.domain.constraints.get(NChooseKConstraint)

        # Build mapping from feature key to tensor column index
        features2idx, _ = self.domain.inputs._get_transform_info(
            self.input_preprocessing_specs,
        )

        # Collect all NChooseK feature indices and their constraint info
        nchoosek_feature_indices: list[int] = []
        for c in nchoosek_constraints:
            assert isinstance(c, NChooseKConstraint)
            for feat_key in c.features:
                for idx in features2idx[feat_key]:
                    if idx not in nchoosek_feature_indices:
                        nchoosek_feature_indices.append(idx)

        # Transform candidates to tensor
        transformed = self.domain.inputs.transform(
            candidates, self.input_preprocessing_specs
        )
        X = torch.from_numpy(transformed.values).to(**tkwargs)  # (q, d)

        q = X.shape[0]

        for i in range(q):
            # Pruning loop for candidate i
            remaining_indices = [
                idx for idx in nchoosek_feature_indices if X[i, idx].abs() > 1e-6
            ]

            while not self._nchoosek_fulfilled_tensor(
                X[i], nchoosek_constraints, features2idx
            ):
                if len(remaining_indices) == 0:
                    break

                # Compute base AF value from already-pruned candidates[0:i]
                if i > 0:
                    base_X = X[:i].unsqueeze(0)  # (1, i, d)
                    with torch.no_grad():
                        base_af_val = acqf(base_X)
                else:
                    base_af_val = torch.tensor(0.0, **tkwargs)

                # Compute dense AF value: [pruned_0..i-1, current_candidate_i]
                dense_X = X[: i + 1].unsqueeze(0)  # (1, i+1, d)
                with torch.no_grad():
                    dense_af_val = acqf(dense_X)
                dense_incremental = (dense_af_val - base_af_val).clamp_min(0)

                # Create variants with each remaining dimension zeroed
                n_remaining = len(remaining_indices)
                variants = X[i].unsqueeze(0).expand(n_remaining, -1).clone()
                for k, idx in enumerate(remaining_indices):
                    variants[k, idx] = 0.0

                # Evaluate AF on [pruned_0..i-1, variant_k] for each variant
                if i > 0:
                    # (n_remaining, i+1, d)
                    prefix = X[:i].unsqueeze(0).expand(n_remaining, -1, -1)
                    eval_X = torch.cat([prefix, variants.unsqueeze(1)], dim=1)
                else:
                    eval_X = variants.unsqueeze(1)  # (n_remaining, 1, d)

                with torch.no_grad():
                    variant_af_vals = acqf(eval_X)  # (n_remaining,)

                # Compute AF reduction for each variant
                if dense_incremental > 0:
                    variant_incremental = (variant_af_vals - base_af_val).clamp_min(0)
                    af_reduction = (
                        dense_incremental - variant_incremental
                    ) / dense_incremental
                else:
                    af_reduction = torch.zeros(n_remaining, **tkwargs)

                # Select dimension with smallest AF reduction
                min_idx = af_reduction.argmin().item()
                best_dim = remaining_indices[min_idx]

                # Zero the selected dimension
                X[i, best_dim] = 0.0
                remaining_indices.remove(best_dim)

        # Convert back to dataframe
        df_result = pd.DataFrame(
            data=X.detach().cpu().numpy(),
            columns=transformed.columns,
            index=candidates.index,
        )
        df_result = self.domain.inputs.inverse_transform(
            df_result, self.input_preprocessing_specs
        )
        return df_result

    @staticmethod
    def _nchoosek_fulfilled_tensor(
        x: Tensor,
        nchoosek_constraints: list,
        features2idx: Dict[str, Tuple],
        tol: float = 1e-6,
    ) -> bool:
        """Check if all NChooseK constraints are fulfilled for a single candidate tensor."""
        for c in nchoosek_constraints:
            assert isinstance(c, NChooseKConstraint)
            indices = []
            for feat_key in c.features:
                indices.extend(features2idx[feat_key])
            count = (x[indices].abs() > tol).sum().item()
            if count > c.max_count:
                return False
            if count < c.min_count and not (c.none_also_valid and count == 0):
                return False
        return True

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
        """

        Returns:
            X_train (Tensor): Tensor of shape (n, d) with n training points and d input dimensions.
            X_pending (Tensor | None): Tensor of shape (m, d) with m pending points

        """
        assert self.experiments is not None
        experiments = self.domain.outputs.preprocess_experiments_all_valid_outputs(
            self.experiments,
        )

        clean_experiments = experiments.drop_duplicates(
            subset=[var.key for var in self.domain.inputs.get(Input)],
            keep="first",
            inplace=False,
        )
        if not self.include_infeasible_exps_in_acqf_calc:
            # we should only provide those experiments to the acqf builder in which all
            # input constraints are fulfilled, output constraints are handled directly
            # in botorch
            fulfilled_experiments = clean_experiments[
                self.domain.is_fulfilled(clean_experiments)
            ].copy()
            if len(fulfilled_experiments) == 0:
                warnings.warn(
                    "No valid and feasible experiments are available for setting up the acquisition function. Check your constraints.",
                    RuntimeWarning,
                )
            else:
                clean_experiments = fulfilled_experiments
        # else:
        #     clean_experiments = clean_experiments.copy()

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
            objective=objective,
        )
