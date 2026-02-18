# Changelog
All notable changes to BoFire will be documented in this file starting from February 2026.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Pragmatic Versioning](https://github.com/experimental-design/bofire?tab=readme-ov-file#versioning).

## [Unreleased]

### Added

- `PFNSurrogate` - Prior-data Fitted Networks (PFN) surrogate model for Bayesian optimization using pre-trained transformers from the `pfns4bo` library. Includes support for both univariate and multivariate outputs with custom serialization for outcome transforms.
- `CloneFeatures` engineered feature, that can be used to create a copy of a set of features, this can be useful if one wants to further process features differently (different scalers, different kernels etc.)
- Explicit Interaction features (like `x_1 * x_2`) for botorch based surrogates via the engineered features mechanism.
- Support for custom formulas including discrete and categorical features in the DoE module.
- Support for pandas 3.0

### Changed

- **Breaking**: For all botorch surrogate that are trainable, the `scaler` keyword used on defining how to scale the inputs before entering the actual model/kernel, do not expect anymore an enum but instance of a `Scaler` class like `Normalize` or `Standardize`. Via this, it can be controlled on which features the scaler should operate.
- Static type checking was migrated from `pyright` to `ty`.

### Fixed

- Flaky tests in the test pipeline
