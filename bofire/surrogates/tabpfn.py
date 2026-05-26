"""TabPFN surrogate (regression).

TabPFN is a pre-trained tabular meta-learner — there is no parameter-fitting
step, just an in-context forward pass over `(X_train, y_train, X_test)` that
emits logits over a bar distribution.

This module is structured top-down: low-level adapter, then the botorch
``Model`` wrapper, then the BoFire surrogate.
"""

from __future__ import annotations

import inspect
from functools import lru_cache
from typing import Any, Optional

import torch
from botorch.exceptions.errors import UnsupportedError
from botorch.models.model import Model
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.posteriors import Posterior
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.utils.transforms import match_batch_shape
from gpytorch.distributions import MultivariateNormal
from torch import Tensor, nn

from bofire.data_models.enum import OutputFilteringEnum
from bofire.data_models.surrogates.api import TabPFNSurrogate as DataModel
from bofire.surrogates.botorch import TrainableBotorchSurrogate


@lru_cache(maxsize=4)
def _load_tabpfn(
    version: str, which: str, checkpoint: Optional[str], device: str, use_kv_cache: bool
):
    """Load a TabPFN architecture + bar-distribution criterion.

    Cached because the checkpoint is ~50 MB and a BO loop calls
    ``_fit_botorch`` once per iteration. The criterion's ``borders`` are in
    z-space (the model was trained on z-normed targets); rescaling back to
    raw-y space is the outcome transform's job.
    """
    try:
        from tabpfn.model_loading import load_model_criterion_config
    except ImportError as e:
        raise ImportError(
            "TabPFN is not installed. Install with `pip install bofire[tabpfn]`."
        ) from e
    models, criterion, _, _ = load_model_criterion_config(
        model_path=checkpoint,
        version=version,
        which=which,
        check_bar_distribution_criterion=True,
        cache_trainset_representation=use_kv_cache,
        download_if_not_exists=True,
    )
    arch = models[0].to(device).eval()
    criterion = criterion.to(device)
    return arch, criterion


class TabPFNAdapter(nn.Module):
    """Pfns4bo-shaped wrapper around a TabPFN architecture.

    TabPFN's `arch.forward` takes ``(x, y)`` with ``x = cat([X_train, X_test])``
    concatenated along the sequence dimension; ``single_eval_pos`` is inferred
    from ``y.shape[0]``. We expose a ``forward(x, y, test_x)`` API matching
    pfns4bo so the rest of the wrapper stays simple.

    Version-specific kwargs (v2.x: ``style``/``data_dags``; v3:
    ``kv_cache``/``x_is_test_only``) are filtered via ``inspect.signature``.
    """

    def __init__(
        self,
        version: str = "v3",
        which: str = "regressor",
        checkpoint: Optional[str] = None,
        use_kv_cache: bool = True,
        device: str = "cpu",
    ):
        super().__init__()
        self.arch, self.criterion = _load_tabpfn(
            version=version,
            which=which,
            checkpoint=checkpoint,
            device=device,
            use_kv_cache=use_kv_cache,
        )
        self._fwd_params = set(inspect.signature(self.arch.forward).parameters)

    @property
    def borders(self) -> Tensor:
        return self.criterion.borders

    def forward(self, x: Tensor, y: Tensor, test_x: Tensor) -> Tensor:
        """Run the architecture and return logits for the test rows.

        Args:
            x: training features, shape ``(n, B, d)``.
            y: training targets, shape ``(n, B, 1)``.
            test_x: query features, shape ``(q, B, d)``.

        Returns:
            Logits, shape ``(q, B, num_buckets)``. TabPFN's architecture
            returns logits for the test segment only — the training segment
            is consumed for conditioning and not output.
        """
        # TabPFN expects float32 throughout.
        x = x.float()
        y = y.float()
        test_x = test_x.float()
        # Sequence-dim concat; single_eval_pos inferred from y.shape[0].
        x_full = torch.cat([x, test_x], dim=0)
        batch = x_full.shape[1]
        kwargs: dict[str, Any] = {
            "x": x_full,
            "y": y,
            "only_return_standard_out": True,
            "task_type": "regression",
            # Passing None triggers different encoder behaviour; explicit empty
            # lists per-batch are what TabPFN expects when there are no
            # categoricals after BoFire's preprocessing pipeline.
            "categorical_inds": [[] for _ in range(batch)],
        }
        kwargs = {k: v for k, v in kwargs.items() if k in self._fwd_params}
        return self.arch(**kwargs)


class TabPFNModel(Model):
    """botorch-compatible wrapper around a TabPFNAdapter.

    Departures from ``botorch_community.PFNModel``:
      - Accepts ``outcome_transform`` and rescales the posterior accordingly.
      - Selectable ``posterior_type`` (Gaussian via ``GPyTorchPosterior``,
        or ``BoundedRiemannPosterior``).
      - Single-output only; raises on ``output_indices``; ignores
        ``observation_noise`` (PFN has no noise model).
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        adapter: TabPFNAdapter,
        input_transform: Optional[InputTransform] = None,
        outcome_transform: Optional[OutcomeTransform] = None,
        posterior_type: str = "gaussian",
    ):
        super().__init__()
        if train_X.dim() != 2:
            raise UnsupportedError("train_X must be 2-dimensional.")
        if train_Y.dim() != 2 or train_Y.shape[-1] != 1:
            raise UnsupportedError("Only single-output regression is supported.")
        if train_X.shape[0] != train_Y.shape[0]:
            raise UnsupportedError("train_X and train_Y must have the same n.")

        with torch.no_grad():
            self.transformed_X = self.transform_inputs(
                X=train_X, input_transform=input_transform
            )
        if outcome_transform is not None:
            # Fits Standardize stats lazily on first call.
            train_Y_z, _ = outcome_transform(train_Y)
            self.outcome_transform = outcome_transform
        else:
            train_Y_z = train_Y
        self.train_X = train_X
        self.train_Y = train_Y_z
        self.adapter = adapter
        self.posterior_type = posterior_type
        if input_transform is not None:
            self.input_transform = input_transform

    @property
    def num_outputs(self) -> int:
        return 1

    def _prepare(self, X: Tensor) -> tuple[Tensor, Tensor, Tensor, torch.Size]:
        """Shape juggling cribbed from ``PFNModel._prepare_data``.

        X comes in as ``(b?, q?, d)``; pad to ``(b, q, d)``, apply
        ``input_transform``, and broadcast the cached train tensors to the
        same leading batch shape.
        """
        orig_X_shape = X.shape
        if X.dim() > 3:
            raise UnsupportedError(f"X must be at most 3-d, got {X.shape}.")
        while X.dim() < 3:
            X = X.unsqueeze(0)
        X = self.transform_inputs(X)
        train_X = match_batch_shape(self.transformed_X, X)
        train_Y = match_batch_shape(self.train_Y, X)
        return X, train_X, train_Y, orig_X_shape

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[list[int]] = None,
        observation_noise: bool | Tensor = False,
        posterior_transform=None,
    ) -> Posterior:
        if output_indices is not None:
            raise UnsupportedError("output_indices is not supported for TabPFN.")
        # observation_noise silently ignored — BoFire's _predict always passes
        # True and PFN has no noise model. Matches upstream PFNModel behaviour.

        X_t, train_X_b, train_Y_b, orig_shape = self._prepare(X)
        # Adapter wants (seq, batch, ...) layout. Keep logits in the
        # architecture's native dtype (float32) — the bar distribution's
        # internal buffers are float32, and mixing dtypes through
        # ``mean_of_square`` fails inside TabPFN.
        logits = self.adapter(
            x=train_X_b.transpose(0, 1),
            y=train_Y_b.transpose(0, 1),
            test_x=X_t.transpose(0, 1),
        ).transpose(0, 1)  # (b, q, num_buckets), float32

        if self.posterior_type == "gaussian":
            post = self._gaussian_posterior(logits, orig_shape)
        elif self.posterior_type == "riemann":
            post = self._riemann_posterior(logits, orig_shape)
        else:
            raise ValueError(f"Unknown posterior_type: {self.posterior_type}")

        if posterior_transform is not None:
            return posterior_transform(post)
        return post

    def _bardist_stats(self, logits: Tensor) -> tuple[Tensor, Tensor]:
        """Compute (mean, variance) in z-space from logits via the bar distribution.

        ``bardist.mean/variance`` apply softmax to logits internally — pass
        raw logits, never ``log(softmax(...))``.

        ``FullSupportBarDistribution.mean`` moves logits to its own device but
        ``mean_of_square`` (called by ``variance``) does not — we align once
        and convert results back. Bardist buffers are float32 so we keep
        logits float32 throughout and let callers cast to their preferred
        dtype.
        """
        bardist = self.adapter.criterion
        bd_device = bardist.borders.device
        logits_bd = logits.to(device=bd_device, dtype=bardist.borders.dtype)
        mean = bardist.mean(logits_bd).to(logits.device)
        var = bardist.variance(logits_bd).to(logits.device)
        return mean, var

    def _gaussian_posterior(
        self, logits: Tensor, orig_shape: torch.Size
    ) -> GPyTorchPosterior:
        out_dtype = self.train_X.dtype
        z_mean, z_var = self._bardist_stats(logits)  # (b, q), bardist dtype
        z_mean = z_mean.reshape(*orig_shape[:-1]).to(out_dtype)
        z_var = z_var.reshape(*orig_shape[:-1]).to(out_dtype)
        cov = torch.diag_embed(z_var.clamp(min=1e-12))
        mvn = MultivariateNormal(z_mean, cov)
        post = GPyTorchPosterior(distribution=mvn)
        if hasattr(self, "outcome_transform"):
            post = self.outcome_transform.untransform_posterior(post)
        return post

    def _riemann_posterior(self, logits: Tensor, orig_shape: torch.Size):
        # BoundedRiemannPosterior is not a GPyTorchPosterior, so
        # Standardize.untransform_posterior cannot handle it — we rescale
        # the borders manually from the transform's fitted (means, stdvs).
        from botorch_community.posteriors.riemann import BoundedRiemannPosterior

        out_dtype = self.train_X.dtype
        probs = logits.softmax(dim=-1).reshape(*orig_shape[:-1], -1).to(out_dtype)
        borders = self._raw_borders(self.adapter.borders.to(out_dtype))
        return BoundedRiemannPosterior(borders=borders, probabilities=probs)

    def _raw_borders(self, z_borders: Tensor) -> Tensor:
        if not hasattr(self, "outcome_transform"):
            return z_borders
        # Standardize stores (1, m) buffers; m == 1 here.
        stdvs = self.outcome_transform.stdvs.flatten().to(z_borders)
        means = self.outcome_transform.means.flatten().to(z_borders)
        return z_borders * stdvs + means


class TabPFNSurrogate(TrainableBotorchSurrogate):
    def __init__(self, data_model: DataModel, **kwargs):
        self.tabpfn_version = data_model.tabpfn_version
        self.posterior_type = data_model.posterior_type
        self.use_kv_cache = data_model.use_kv_cache
        self.device = data_model.device
        self.checkpoint_path = data_model.checkpoint_path
        super().__init__(data_model=data_model, **kwargs)

    model: Optional[TabPFNModel] = None
    _output_filtering: OutputFilteringEnum = OutputFilteringEnum.ALL

    def _fit_botorch(
        self,
        tX: torch.Tensor,
        tY: torch.Tensor,
        input_transform: Optional[InputTransform] = None,
        outcome_transform: Optional[OutcomeTransform] = None,
        **kwargs,
    ):
        adapter = TabPFNAdapter(
            version=self.tabpfn_version,
            which="regressor",
            checkpoint=self.checkpoint_path,
            use_kv_cache=self.use_kv_cache,
            device=self.device,
        )
        self.model = TabPFNModel(
            train_X=tX,
            train_Y=tY,
            adapter=adapter,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
            posterior_type=self.posterior_type,
        )
