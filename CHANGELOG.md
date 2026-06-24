# Changelog
All notable changes to BoFire will be documented in this file starting from February 2026.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Pragmatic Versioning](https://github.com/experimental-design/bofire?tab=readme-ov-file#versioning).

## [Unreleased]

### Added

### Changed

### Fixed
- Pin `pydantic-ai<2.0.0` to avoid breaking API changes introduced in pydantic-ai 2.0.0 (`output_retries` removed, `OpenAIModel` moved).

## [0.4.1] - 2026-06-16

### Added
- Support for dimensionality scaled gamma priors.

### Changed
- **Breaking**: Botorch >= 0.18.1

## [0.4.0] - 2026-06-08

### Added

- Support for python 3.14
- `CloneFeatures` engineered feature, that can be used to create a copy of a set of features, this can be useful if one wants to further process features differently (different scalers, different kernels etc.)
- Explicit Interaction features (like `x_1 * x_2`) for botorch based surrogates via the engineered features mechanism.
- Support for custom formulas including discrete and categorical features in the DoE module.
- Support for pandas 3.0
- `WeightedMeanFeature` and `MolecularWeightedMeanFeature` engineered features for weighted-mean behavior.
- `plot_gp_slice_plotly` now supports fixed input features that can be a mix of `ContinuousInput` and `CategoricalInput` (with string categorical fixed values).
- Configurable `noise_constraint` support for GP-based surrogates (`SingleTaskGP`, `MixedSingleTaskGP`, `TanimotoGP`, `MultiTaskGP`, and `RobustSingleTaskGP`) and corresponding linear/polynomial wrappers.
- Support for the `PathwiseThompsonSampling` acquisition function.
- Optional `initial_value` field on the `GreaterThan`, `LessThan`, and `Positive` prior constraint data models (already present on `Interval`), letting users opt-in to a warm-start of the constrained gpytorch parameter at construction time.
- Generalized NChooseK constraint support in DoE: `min_count > 0` is now supported, non-zero lower bounds (`lb > 0`) are allowed for NChooseK features, overlapping NChooseK constraints (shared features) are handled via incremental pairwise merge with consistency filtering, and `nchoosek_constraints_as_bounds` generates deactivation patterns for all activity levels `k ∈ [min_count, max_count]`.
- `PairwiseGPSurrogate`, a Gaussian process surrogate that learns a latent utility function from pairwise preference/comparison data, wrapping BoTorch's `PairwiseGP`. The pairwise likelihood is selectable via `likelihood="probit"` (default) or `"logit"`.
- `SmoothedBoxPrior` prior, and a concrete instantiable `Interval` prior constraint.
- Aggregation of duplicated experiments in the `cross_validate` method of trainable surrogates to avoid data leakage, controlled via the `aggregate` boolean flag, default `False`.
- `ExactWassersteinKernel`: exact W1/W2 Wasserstein-distance kernel computed over the union of unique x-breakpoints (vs. the interpolated `WassersteinKernel`). Chunked over the x-union to keep memory bounded on large problems. ([#750](https://github.com/experimental-design/bofire/pull/750))
- Support for multi-output, multi-fidelity Bayesian optimization (MOMF), including a split of `TaskInput` into continuous and categorical task variants.
- LLM-based strategy with provider infrastructure for agentic optimization workflows, plus `context` fields and `to_description()` / `to_pydantic_field()` helpers on data models to support natural-language descriptions of domains.
- `Log` and `ChainedOutputTransform` output transforms for surrogate models.
- `ContinuousMolecularInput` feature and decorrelated molecular features.
- `SphericalLinearKernel`.
- `IndexKernel` and `PositiveIndexKernel` for categorical/task indices.
- `EnsembleMapSaasSingleTaskGPSurrogate`, an ensemble MAP variant of the SAAS single-task GP.
- **Breaking**: `InterpolateFeature` engineered feature, replacing the dedicated `PiecewiseLinearGPSurrogate` with a more flexible engineered-feature based approach.
- Public `register()` functions/decorators that let downstream packages register custom strategies, surrogates, kernels, priors, and engineered features into BoFire's type unions.
- Exposed sampling method, `n_burnin`, and `n_thinning` parameters on `RandomStrategy` for finer control over constrained sampling.
- `allow_zero` flag for NChooseK sampling in `RandomStrategy`.

### Changed

- **Breaking**: Entmoot >= 2.1.1
- **Breaking**: Botorch >= 0.18.0
- **Breaking**: Python >= 3.11
- **Breaking**: For all botorch surrogate that are trainable, the `scaler` keyword used on defining how to scale the inputs before entering the actual model/kernel, do not expect anymore an enum but instance of a `Scaler` class like `Normalize` or `Standardize`. Via this, it can be controlled on which features the scaler should operate.
- **Breaking**: Switched the cheminformatics dependency from the unmaintained `mordred` to `mordredcommunity`.
- Interval.initial_value` (covering `NonTransformedInterval` and `LogTransformedInterval`) is now `Optional[PositiveFloat] = None` — previously a required `PositiveFloat`. This matches gpytorch's and botorch's contract: a `None` initial value means no warm-start at registration time. Existing code that sets `initial_value` keeps working unchanged.
- `noise_constraint` default on the GP surrogates (`SingleTaskGP`, `MultiTaskGP`, `MixedSingleTaskGP`, `TanimotoGP`, `RobustSingleTaskGP`, `LinearSurrogate`, `PolynomialSurrogate`) changed from `None` to `GreaterThan(lower_bound=1e-4)`, mirroring BoTorch's `SingleTaskGP` factory default. `None` is still accepted, so previously-serialised specs continue to round-trip.
- `MultiTaskGPSurrogate`'s default kernel and noise prior now match BoTorch's `MultiTaskGP` default (`RBFKernel(ard=True)` with the HVARFNER lengthscale prior, `LogNormalPrior(-4, 1)` noise prior) and align with `SingleTaskGPSurrogate`. Previously defaulted to `MaternKernel(nu=2.5)` with `GammaPrior(3.0, 6.0)` lengthscale prior and `GammaPrior(1.1, 0.05)` noise prior.
- `MultiTaskGPHyperconfig.prior` categories changed from `["mbo", "botorch"]` to `["mbo", "threesix", "hvarfner"]`, matching `SingleTaskGPHyperconfig`. The old `"botorch"` label mapped to the THREESIX prior; the new `"hvarfner"` label uses BoTorch's current default HVARFNER prior.
- Static type checking was migrated from `pyright` to `ty`.
- Refactored weighted engineered-feature surrogate mapping to share implementation across weighted sum/mean and molecular weighted sum/mean.
- Objective bounds validation for `IdentityObjective`-based objectives is now strict (`lower < upper`) to prevent degenerate normalization ranges.
- `WassersteinKernel` is now a `FeatureSpecificKernel` and accepts an optional `lengthscale_constraint`. ([#750](https://github.com/experimental-design/bofire/pull/750))
- Refactored `Any*` type aliases throughout the data models to Pydantic discriminated unions (`Field(discriminator="type")`) for faster validation and clearer error messages.
- `RandomStrategy` NChooseK sampling now scales to large combinatorial spaces via on-demand sampling.
- Pre-compute the Tanimoto similarity matrix inside the Tanimoto GP for substantial speed-ups, with a configurable calculation mode.

### Removed

- `PiecewiseLinearGPSurrogate` and supporting code in `torch_tools.py`. Equivalent behavior is now available via the `InterpolateFeature` engineered feature with a standard GP surrogate. ([#750](https://github.com/experimental-design/bofire/pull/750))
- Removed the bundled `bofire_theme.mp3` audio file from the base install.

### Fixed

- `Domain.aggregate_by_duplicates` now preserves and aggregates `CategoricalOutput`s by majority vote, breaking ties randomly; `cross_validate(..., aggregate=True)` passes through `random_state` to make tied categorical aggregation reproducible.
- `noise_prior` and `noise_constraint` set on `SingleTaskGP`, `MultiTaskGP`, `TanimotoGP`, and `RobustSingleTaskGP` surrogates now actually influence the GP fit. Previously they were assigned via attribute on `model.likelihood.noise_covar` after model construction, which did not populate gpytorch's `_priors` / `_constraints` registries — so the user-supplied prior was silently ignored by the marginal log-likelihood and the user-supplied constraint's bounds were silently not enforced. ([#762](https://github.com/experimental-design/bofire/issues/762), [#763](https://github.com/experimental-design/bofire/pull/763), [#766](https://github.com/experimental-design/bofire/pull/766))
- Flaky tests in the test pipeline
- Serialization tests now explicitly assert the expected `DeprecationWarning` for deprecated `FactorialStrategy` specs instead of treating it as an unhandled warning.
- Added soft divide in `interp1d` (`torch_tools.py`) to prevent division-by-zero errors on initial candidates that violated inequality constraints. ([#750](https://github.com/experimental-design/bofire/pull/750))
- Fixed bug with discrete variables in the DoE strategy and added handling for adversarial user input.
- Fixed Jacobian/Hessian evaluation for list expressions in `NonlinearConstraint`.
- Fixed `MultiTaskGPSurrogate` serialization that was broken by the new BoTorch default task `BetaPrior` (now passes `task_covar_prior=None`).
- `NonlinearConstraint` evaluation now preserves the original DataFrame index.
- `CategoricalDescriptorInput` now supports being used with only one descriptor.
