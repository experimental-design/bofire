# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Acquisition optimization**
  - Fallback when BoTorch raises `CandidateGenerationError`: retry optimization without nonlinear constraints, then project candidates onto the nonlinear feasible set via `_project_onto_nonlinear_constraints`.
  - For domains with only NChooseK/Product constraints (no nonlinear equality/inequality), use BoTorch’s `gen_batch_initial_conditions` and a `generator` for reproducible initial conditions.
- **Documentation**
  - New Quarto tutorials: `docs/tutorials/advanced_examples/nonlinear_advanced.qmd` (advanced nonlinear constraint examples) and `docs/tutorials/advanced_examples/nonlinear_constraints_maximizing_yield.qmd` (competing reactions with callable constraints). Both listed in the advanced examples index.

### Fixed

- **Nonlinear constraints**
  - Correct handling of nonlinear equality vs inequality when passing constraints to BoTorch and in projection (use domain’s `NonlinearEqualityConstraint` for equality, not the `is_equality` flag from `get_nonlinear_constraints`).
  - Projection and feasible initial-condition generation use the actual variable bounds from the acquisition function instead of a hardcoded `[0, 1]`.
  - When all `NonlinearInequalityConstraint`s use callable expressions, they are no longer passed to BoTorch; custom feasible IC generation is disabled to avoid validation issues; feasibility is enforced via BoFire’s domain validation.
  - Safe handling of `None` from `_get_nonlinear_constraint_setup`: normalize only empty lists to `None` before `len()` (avoids `TypeError` when callable constraints are used).
  - Missing import of `NonlinearInequalityConstraint` in `acqf_optimization.py` (fixes `NameError` in NChooseK-only branch).
- **BotorchStrategy**
  - `has_sufficient_experiments`: exclude interpoint constraints (e.g. `InterpointEqualityConstraint`) from the feasibility check, since they apply to the batch of candidates requested, not to past experiments. Fixes “Not enough experiments” when only interpoint constraints are present.
  - Reindex `feasible_mask` to `valid_experiments.index` before boolean indexing to avoid pandas `IndexingError: Unalignable boolean Series`.
- **InterpointEqualityConstraint**
  - `is_fulfilled` now returns a Series with `index=experiments.index` and the same boolean for all rows (batch feasible or not), so it aligns with other constraints in `Constraints.is_fulfilled`.
- **Tutorial (nonlinear_constraints_basic_usage.py)**
  - When using callable nonlinear constraints, ask with `raise_validation_error=False` and `add_pending=False`, then keep only feasible candidates with a retry loop (up to 10 attempts) so the script completes without `ConstraintNotFulfilledError`.

### Changed

- **Acquisition optimization**
  - Refactored nonlinear constraint and IC setup into `_get_nonlinear_constraint_setup`; added `skip_nonlinear` argument to `_get_arguments_for_optimizer` for the `CandidateGenerationError` fallback path.
  - Removed debug `print` statements from the feasible IC generator.
- **Data models / constraints**
  - Removed DEBUG print block from `NonlinearEqualityConstraint.is_fulfilled` in `bofire/data_models/constraints/nonlinear.py`.
