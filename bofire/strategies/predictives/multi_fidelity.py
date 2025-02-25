import math
from typing import Optional

import numpy as np
import pandas as pd
import torch
from botorch.acquisition import (
    SampleReducingMCAcquisitionFunction,
    qMultiFidelityLowerBoundMaxValueEntropy,
    qMultiFidelityMaxValueEntropy,
)
from botorch.acquisition.objective import MCAcquisitionObjective, PosteriorTransform
from botorch.acquisition.utils import project_to_target_fidelity
from botorch.models.model import Model
from botorch.sampling.base import MCSampler

from bofire.data_models.acquisition_functions.api import qMFVariance
from bofire.data_models.features.api import TaskInput
from bofire.data_models.strategies.predictives.multi_fidelity import (
    MultiFidelityStrategy as DataModel,
)
from bofire.strategies.predictives.sobo import SoboStrategy
from bofire.utils.naming_conventions import get_column_names
from bofire.utils.torch_tools import tkwargs


class qMultiFidelityVariance(SampleReducingMCAcquisitionFunction):
    r"""MC-based Variance Bound.

    Uses a reparameterization to extend UCB to qUCB for q > 1 (See Appendix A
    of [Wilson2017reparam].) Since we only consider the variance, we get the following
    expression.

    `qVariance = E(max(|Y_tilde - mu|))`, where `Y_tilde ~ N(mu, beta pi/2 Sigma)`
    and `f(X)` has distribution `N(mu, Sigma)`.


    """

    def __init__(
        self,
        model: Model,
        beta: float,
        fidelity_thresholds: torch.Tensor,
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCAcquisitionObjective] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
        X_pending: Optional[torch.Tensor] = None,
    ) -> None:
        r"""q-Upper Confidence Bound.

        Args:
            model: A fitted model.
            beta: Controls tradeoff between mean and standard deviation in UCB.
            sampler: The sampler used to draw base samples. See `MCAcquisitionFunction`
                more details.
            objective: The MCAcquisitionObjective under which the samples are
                evaluated. Defaults to `IdentityMCObjective()`.
            posterior_transform: A PosteriorTransform (optional).
            X_pending: A `batch_shape x m x d`-dim Tensor of `m` design points that have
                points that have been submitted for function evaluation but have not yet
                been evaluated. Concatenated into X upon forward call. Copied and set to
                have no gradient.
        """
        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
        )
        self.beta_prime = self._get_beta_prime(beta=beta)
        self.fidelity_thresholds = fidelity_thresholds

    def _get_beta_prime(self, beta: float) -> float:
        return math.sqrt(beta * math.pi / 2)

    def _sample_forward(self, obj: torch.Tensor) -> torch.Tensor:
        r"""Evaluate qMultiFidelityVariance per sample on the candidate set `X`.

        Args:
            obj: A `sample_shape x batch_shape x q`-dim Tensor of MC objective values.

        Returns:
            A `sample_shape x batch_shape x q`-dim Tensor of acquisition values.
        """
        mean = obj.mean(dim=0)
        return self.beta_prime * (obj - mean).abs()

    def forward(self, X: torch.Tensor):
        r"""Compute acquisition values for a batch of a design point with different fidelities.

        Since the acquisition function depends on other fidelities, we need to
        share information across a batch of samples across fidelities. We therefore
        need to override the forward method to handle this.

        Args:
            X: A `batch_shape x q=1 x d`-dim Tensor. X must be ordered from lowest
                to highest fidelity.

        Returns:
            A `batch_shape`-dim Tensor of acquisition values.
        """
        acqf_values = super().forward(X)
        acqf_over_threshold = torch.zeros_like(acqf_values)

        fidelity_threshold_scale = self.model.outcome_transform.stdvs.item()
        fidelity_thresholds = self.fidelity_thresholds * fidelity_threshold_scale
        above_threshold = acqf_values > fidelity_thresholds

        if above_threshold.sum() == 0:
            acqf_over_threshold[-1] = 1.0
        else:
            first_above_threshold = torch.argmax(above_threshold, dim=0)
            acqf_over_threshold[first_above_threshold] = 1.0
        return acqf_over_threshold


def get_mf_acquisition_function(
    acquisition_function_name: str,
    model: Model,
    target_fidelities: dict[int, float],
    # objective: MCAcquisitionObjective,
    # X_observed: Tensor,
    # posterior_transform: Optional[PosteriorTransform] = None,
    # X_pending: Optional[Tensor] = None,
    # constraints: Optional[list[Callable[[Tensor], Tensor]]] = None,
    # eta: Optional[Union[Tensor, float]] = 1e-3,
    # mc_samples: int = 512,
    # seed: Optional[int] = None,
    *,
    beta: Optional[float] = None,
    fidelity_thresholds: Optional[torch.Tensor] = None,
    candidate_set: Optional[torch.Tensor] = None,
):
    """Convenience function for initialiing multi-fidelity acquisition functions.

    Mirrors the signature of botorch.acquisition.factory.get_acquisition_function.
    """

    def project(X):
        return project_to_target_fidelity(X=X, target_fidelities=target_fidelities)

    if acquisition_function_name in ["qMFMES", "qMFGibbon"]:
        if candidate_set is None:
            raise ValueError(
                "`candidate_set` must not be None for qMFMES and qMFGibbon."
            )

    if acquisition_function_name == "qMFMES":
        return qMultiFidelityMaxValueEntropy(
            model=model,
            candidate_set=candidate_set,  # type: ignore
            project=project,
        )

    elif acquisition_function_name == "qMFGibbon":
        return qMultiFidelityLowerBoundMaxValueEntropy(
            model=model,
            candidate_set=candidate_set,  # type: ignore
            project=project,
        )

    elif acquisition_function_name == "qMFVariance":
        if beta is None:
            raise ValueError("`beta` must not be None for qMFVariance.")
        if fidelity_thresholds is None:
            raise ValueError("`fidelity_thresholds` must not be None for qMFVariance.")
        return qMultiFidelityVariance(
            model=model,
            beta=beta,
            fidelity_thresholds=fidelity_thresholds,
        )

    raise NotImplementedError(
        f"Unknown acquisition function {acquisition_function_name}"
    )


class MultiFidelityStrategy(SoboStrategy):
    def __init__(self, data_model: DataModel, **kwargs):
        super().__init__(data_model=data_model, **kwargs)
        self.task_feature_key = self.domain.inputs.get_keys(TaskInput)[0]
        self.fidelity_acquisition_function = data_model.fidelity_acquisition_function

        # ft = data_model.fidelity_thresholds
        # M = len(self.domain.inputs.get_by_key(self.task_feature_key).fidelities)  # type: ignore
        # self.fidelity_thresholds = ft if isinstance(ft, list) else [ft] * M

    def _ask(self, candidate_count: int) -> pd.DataFrame:
        """Generate new candidates (x, m).

        This is a greedy optimization of the acquisition function. We first
        optimize the acqf for the target fidelity to generate a candidate x,
        then select a target fidelity.

        We do this procedure greedily in line with [Folch et al. 2023]. This has
        the advantage of being simpler and faster, as we only need to evaluate
        the fidelity acquisition function M times. It also allows more freedom
        in the choice of design-space acquisition function, as well as enabling a
        more flexible choice of surrogate models.

        Args:
            candidate_count (int): number of candidates to be generated

        Returns:
            pd.DataFrame: DataFrame with candidates (proposed experiments)
        """
        if candidate_count > 1:
            raise NotImplementedError("Batch optimization is not yet implemented")

        self._verify_all_fidelities_observed()

        task_feature: TaskInput = self.domain.inputs.get_by_key(self.task_feature_key)  # type: ignore
        # only optimize the input x on the target fidelity
        # we fix the fidelity by setting all other fidelities to 'not allowed'
        prev_allowed = task_feature.allowed
        task_feature.allowed = [fidelity == 0 for fidelity in task_feature.fidelities]
        x = super()._ask(candidate_count)
        task_feature.allowed = prev_allowed
        fidelity_pred = self._select_fidelity_and_get_predict(x)
        x.update(fidelity_pred)
        return x

    def _select_fidelity_and_get_predict(self, X: pd.DataFrame) -> pd.DataFrame:  # type: ignore
        """Select the fidelity for a given input.

        Uses the variance based approach (see [Kandasamy et al. 2016,
        Folch et al. 2023]) to select the lowest fidelity that has a variance
        exceeding a threshold. If no such fidelity exists, pick the target fidelity

        Args:
            X (pd.DataFrame): optimum input of target fidelity

        Returns:
            pd.DataFrame: selected fidelity and prediction
        """
        fidelity_input: TaskInput = self.domain.inputs.get_by_key(self.task_feature_key)  # type: ignore
        assert self.model is not None and self.experiments is not None
        assert fidelity_input.allowed is not None

        sorted_fidelities = np.argsort(fidelity_input.fidelities)[::-1]
        target_fidelity_idx = sorted_fidelities[-1]
        target_fidelity = fidelity_input.fidelities[target_fidelity_idx]
        num_fidelities = len(fidelity_input.fidelities)
        _, sd_cols = get_column_names(self.domain.outputs)

        fidelity_acqf = get_mf_acquisition_function(
            acquisition_function_name=self.fidelity_acquisition_function.__class__.__name__,
            model=self.model,
            target_fidelities={target_fidelity_idx: float(target_fidelity)},
            beta=(
                self.fidelity_acquisition_function.beta
                if isinstance(self.fidelity_acquisition_function, qMFVariance)
                else 0.2
            ),
        )

        X_fidelity_batched = X.loc[X.index.repeat(num_fidelities)]
        X_fidelity_batched[self.task_feature_key] = np.repeat(
            fidelity_input.categories, len(X)
        )
        # TODO: check that this transform is correct
        X_fidelity_batched = self.domain.inputs.transform(
            experiments=X_fidelity_batched, specs=self.input_preprocessing_specs
        )
        X_fidelity_batched_tensor = torch.from_numpy(X_fidelity_batched.to_numpy()).to(
            **tkwargs
        )
        acqf_values = fidelity_acqf(X_fidelity_batched_tensor)

        chosen_fidelity_idx = int(torch.argmax(acqf_values).item())
        candidate = X_fidelity_batched.iloc[[chosen_fidelity_idx]]
        return candidate

        # for fidelity_idx in sorted_fidelities:
        #     if not fidelity_input.allowed[fidelity_idx]:
        #         continue

        #     m = fidelity_input.fidelities[fidelity_idx]
        #     fidelity_name = fidelity_input.categories[fidelity_idx]

        #     fidelity_threshold_scale = self.model.outcome_transform.stdvs.item()
        #     fidelity_threshold = self.fidelity_thresholds[m] * fidelity_threshold_scale

        #     X_fid = X.assign(**{self.task_feature_key: fidelity_name})
        #     transformed = self.domain.inputs.transform(
        #         experiments=X_fid, specs=self.input_preprocessing_specs
        #     )
        #     pred = self.predict(transformed)

        #     if (pred[sd_cols] > fidelity_threshold).all().all() or m == target_fidelity:
        #         pred[self.task_feature_key] = fidelity_name
        #         return pred

    def _verify_all_fidelities_observed(self) -> None:
        """Get all fidelities that have at least one observation.

        We use this instead of overriding `has_sufficient_experiments` to provide
        a more descriptive error message."""
        assert self.experiments is not None
        observed_fidelities = set(self.experiments[self.task_feature_key].unique())
        allowed_fidelities = set(
            self.domain.inputs.get_by_key(
                self.task_feature_key
            ).get_allowed_categories()  # type: ignore
        )
        missing_fidelities = allowed_fidelities - observed_fidelities
        if missing_fidelities:
            raise ValueError(f"Some tasks have no experiments: {missing_fidelities}")
