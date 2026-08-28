# Changelog
All notable changes to BoFire will be documented in this file starting from February 2026.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Pragmatic Versioning](https://github.com/experimental-design/bofire?tab=readme-ov-file#versioning).

## [Unreleased]

### Added
- Four GP-based convergence criteria for single-objective Bayesian optimization: `UcbLcbRegretBoundCriterion` (UCB-LCB regret bound, Makarova et al. 2022), `ExpMinRegretGapCriterion` (expected minimum simple regret gap, Ishibashi et al. 2023), `LogEipcCriterion` (cost-aware log expected improvement per cost, Xie et al. 2025), and `ProbabilisticRegretBoundCriterion` (probabilistic regret bounds via a Clopper-Pearson sequential test over GP sample paths, Wilson 2024).
- Single-objective benchmark functions used in the stopping-criteria literature.
- Any input feature can carry a `Descriptors` block — numeric `columns` and/or a SMILES `structure` column — via `descriptors=Descriptors(...)`. This includes `DiscreteInput`, which had no descriptor support before, and allows a single categorical to carry handcrafted columns *and* structures at once, which the previous class hierarchy could not express.
- Categorical encodings are first-class data models (`OneHotEncoding`, `OrdinalEncoding`, `DescriptorEncoding`), chosen per surrogate via `categorical_encodings` and extensible like kernels or priors.
- Data model fields are documented in `Field(description=...)`, so the prose now reaches `model_json_schema()` where users and LLM agents configuring BoFire can read it. `tests/bofire/data_models/test_documentation.py` enforces a description on every field and a docstring on every class, with an allowlist of the models that predate the convention that may only shrink. The `constraints` package is migrated; the remaining packages follow.

### Deprecated
- The non-log acquisition functions `qEI`, `qNEI`, `qEHVI` and `qNEHVI` now emit a `DeprecationWarning` on construction and will be removed in a future release. Use `qLogEI`, `qLogNEI`, `qLogEHVI` and `qLogNEHVI` instead: they optimize the same quantity in log space, which keeps a usable gradient where the plain formulations underflow to zero.

### Changed
- **Breaking**: the outlier-detection layer is **removed**, with no compatibility shim. Gone are the packages `bofire.outlier_detection` and `bofire.data_models.outlier_detection` (`OutlierDetection`, `IterativeTrimming`, `OutlierDetections`) and the `outlier_detection_specs`, `min_experiments_before_outlier_check` and `frequency_check` fields on `BotorchStrategy` — serialized strategies carrying those fields no longer load. Use `RobustSingleTaskGPSurrogate`, which learns a data-point specific noise level and so handles outliers inside the BO loop instead of trimming them in a pre-fit pass.
- **Breaking**: descriptor data now lives on the feature and the encoding choice on the surrogate. The classes and the enum that fused those two concerns are **removed**, with no compatibility shim — old serialized domains containing them no longer load. Migrate as follows (note that `values` was row-per-category while `columns` is column-wise, so the table is transposed):

  | removed | replacement |
  |---|---|
  | `CategoricalDescriptorInput(categories=C, descriptors=[n…], values=[[row]…])` | `CategoricalInput(categories=C, descriptors=Descriptors(columns={n: [row[j] for row in values]}))` |
  | `ContinuousDescriptorInput(descriptors=[n…], values=[v…])` | `ContinuousInput(descriptors=Descriptors(columns={n: [v]}))` |
  | `CategoricalMolecularInput(categories=S)` | `CategoricalInput(categories=S, descriptors=Descriptors(structure=S))` |
  | `ContinuousMolecularInput(molecule=m)` | `ContinuousInput(descriptors=Descriptors(structure=[m]))` |
  | `WeightedMeanFeature(…)` | `WeightedSumFeature(…, normalize=True)` |
  | `MolecularWeightedSumFeature(molfeatures=g)` | `WeightedSumFeature(columns=[], generators=[g])` |
  | `MolecularWeightedMeanFeature(molfeatures=g)` | `WeightedSumFeature(columns=[], generators=[g], normalize=True)` |
  | `CategoricalEncodingEnum.ONE_HOT` / `.ORDINAL` / `.DUMMY` | `OneHotEncoding()` / `OrdinalEncoding()` / `OneHotEncoding(drop_first=True)` |
  | `CategoricalEncodingEnum.DESCRIPTOR` | `DescriptorEncoding()` |
  | a bare generator as an encoding, e.g. `{"x": Fingerprints()}` | `{"x": DescriptorEncoding(columns=[], generators=[Fingerprints()])}` |
  | `input_preprocessing_specs=…` on a surrogate | removed; derived from `inputs`. Use `categorical_encodings` |
  | `CategoricalDescriptorInput.from_df` / `to_df` | build directly: `CategoricalInput(key=…, categories=list(df.index), descriptors=Descriptors(columns=df.to_dict("list")))` |
  | `bofire.data_models.molfeatures` | `bofire.data_models.descriptor_generators` |
  | `MolFeatures` / `AnyMolFeatures` | `DescriptorGenerator` / `AnyDescriptorGenerator` |

- **Breaking**: transformed column order changes for domains that mix descriptor- or structure-carrying features with plain ones. The removed classes occupied `order_id`s 2/4/5/6, interleaved among the survivors; their features now sort as plain `CategoricalInput`/`ContinuousInput` (7/1). Feature *values* are unaffected — only the column positions. Code that indexes transformed tensors positionally should be rechecked; code that goes through `Inputs.get_feature_indices` / `_get_transform_info` needs no change.
- **Breaking**: the default encoding for a categorical, and hence the default surrogate, is chosen from the descriptor data the feature carries rather than from its class. A categorical with numeric columns defaults to `DescriptorEncoding`; one with a structure additionally gets `Fingerprints`; one with neither falls back to one-hot (ordinal where the surrogate requires it). In particular a descriptor-carrying categorical no longer selects `MixedSingleTaskGPSurrogate`.
- **Breaking**: `CategoricalTaskInput` and `ContinuousTaskInput` declare `descriptors: None`, so they now serialize a `"descriptors": null` key and reject descriptor data at the type level — a task input is an index, not a described entity.
- **Breaking**: `LinearDeterministicSurrogate` rejects engineered features with `filter_descriptors=True`. A linear model binds one coefficient per column, so the width must follow from the configuration rather than the data.
- LLM field descriptions report descriptor data uniformly for every feature type: a prefix stating what the feature is and its range or options, then the data — `Categorical, allowed: [...] — descriptors per category: {...} — structure: [...]`, `Continuous, bounds [...] — descriptors: {...} — structure: CCO`. Previously the kind was announced in the prefix (`Categorical with descriptors`, `Continuous molecular (SMILES: CCO)`) and only the two descriptor-carrying classes emitted anything.

### Fixed
- **Descriptor widths no longer depend on evaluation order.** `MolFeatures.get_descriptor_names()` returned the filtered list only once `remove_correlated_descriptors()` had run and mutated the model, so with `filter_descriptors` defaulting to `True` the width reported for an engineered feature could be the *unfiltered* count depending on call order. Widths are now derived on demand from the same block the matrix is built from.

## [0.5.0] - 2026-08-11 - BREAKING

Note that this release has two tags `v0.4.2` and `v0.5.0`. It should have been called `v0.5.0`, though, due to a breaking change. The breaking change has been realized too late, hence we have 2 tags on this commit. Further, the two versions in Pypi 0.4.2 and 0.5.0 are identical.

### Added
- **BREAKING**: needs at least Pandas 3 due to standardized write-on-copy-behaviour, see https://github.com/experimental-design/bofire/pull/791.
- Convergence criteria as a first-class, user-extensible family (`bofire.strategies.convergence_criteria` and `bofire.data_models.strategies.convergence_criteria`). Predictive strategies expose a `convergence_criterion` field and a `has_converged()` method, with `ObjectiveImprovementCriterion` (best-objective stagnation over a lookback window) and `ProposalDeviationCriterion` (normalized deviation between consecutive proposals) as  first simple built-in criteria. Custom criteria can be registered via `bofire.strategies.convergence_criteria.api.register`. Stepwise strategies can advance on convergence via the `StrategyHasConvergedCondition`.
- `FractionalFactorialStrategy.get_required_number_of_experiments()` to return the exact number of experiments generated by the strategy for continuous, mixed, categorical/discrete-only, and blocked designs, avoiding underestimation in mixed full-factorial cases.
- Fix `allow_zero` + categorical variables crashing optimization of the acquisition function in case of batch proposals, see https://github.com/experimental-design/bofire/pull/796.
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
