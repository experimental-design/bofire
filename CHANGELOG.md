# Changelog
All notable changes to BoFire will be documented in this file starting from February 2026.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Pragmatic Versioning](https://github.com/experimental-design/bofire?tab=readme-ov-file#versioning).

## [Unreleased]

### Added

- Support for python 3.14
- `CloneFeatures` engineered feature, that can be used to create a copy of a set of features, this can be useful if one wants to further process features differently (different scalers, different kernels etc.)
- Explicit Interaction features (like `x_1 * x_2`) for botorch based surrogates via the engineered features mechanism.
- Support for custom formulas including discrete and categorical features in the DoE module.
- Support for pandas 3.0
- `WeightedMeanFeature` and `MolecularWeightedMeanFeature` engineered features for weighted-mean behavior.
- `plot_gp_slice_plotly` now supports fixed input features that can be a mix of `ContinuousInput` and `CategoricalInput` (with string categorical fixed values).
- Configurable `noise_constraint` support for GP-based surrogates (`SingleTaskGP`, `MixedSingleTaskGP`, `TanimotoGP`, and `MultiTaskGP`) and corresponding linear/polynomial wrappers.
- Generalized NChooseK constraint support in DoE: `min_count > 0` is now supported, non-zero lower bounds (`lb > 0`) are allowed for NChooseK features, and `nchoosek_constraints_as_bounds` generates deactivation patterns for all activity levels `k ∈ [min_count, max_count]`. When `min_count=0`, the all-zero (fully inactive) pattern is now included naturally. `none_also_valid=True` with `min_count > 0` explicitly adds the all-zero pattern.

### Changed

- **Breaking**: For all botorch surrogate that are trainable, the `scaler` keyword used on defining how to scale the inputs before entering the actual model/kernel, do not expect anymore an enum but instance of a `Scaler` class like `Normalize` or `Standardize`. Via this, it can be controlled on which features the scaler should operate.
- Entmoot >=2.1.1
- Static type checking was migrated from `pyright` to `ty`.
- Refactored weighted engineered-feature surrogate mapping to share implementation across weighted sum/mean and molecular weighted sum/mean.

### Fixed

- Flaky tests in the test pipeline
