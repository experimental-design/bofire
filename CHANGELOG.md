# Changelog
All notable changes to BoFire will be documented in this file starting from February 2026.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Pragmatic Versioning](https://github.com/experimental-design/bofire?tab=readme-ov-file#versioning).

## [Unreleased]

### Added

- Explicit Interaction features (like `x_1 * x_2`) for botorch based surrogates via the engineered features mechanism.
- Support for custom formulas including discrete and categorical features in the DoE module.
- Support for pandas 3.0

### Changed

- Static type checking was migrated from `pyright` to `ty`.

### Fixed

- Flaky tests in the test pipeline
