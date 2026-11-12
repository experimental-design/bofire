# Issue #842: change-by-change review guide

## Scope and comparison

This guide covers the fix relative to base commit `64a46471`, including the
changes in `99c7aac6` and subsequent review follow-ups. It is intended to help
review the pull request; line links refer to the current source.

The originating issue is [#842](https://github.com/experimental-design/bofire/issues/842).
The central example has four mixture components bounded by `[0, 0.6]`, a sum of
one, and at most two active components. A singleton cannot supply the required
sum; a pair can. Pattern generation must not assign impossible singleton patterns
to experiments.

Links below open the changed source files at the relevant line. Line numbers
describe this snapshot and can move after further edits. This is an explanation
and review aid, not a claim that every possible interaction has been certified.

## Suggested review order

1. DoE filter and where it is called.
2. RandomStrategy's fixed-constraint reduction.
3. RandomStrategy's subset retries and sample replenishment.
4. Tensor precision change.
5. Tests, performance measurements, and user-facing documentation.

## 1. DoE: reject impossible patterns before assigning them

**File:** [bofire/strategies/doe/utils.py](bofire/strategies/doe/utils.py#L700)

| Changed location | What changed | Why it belongs here |
|---|---|---|
| [New feasibility helper](bofire/strategies/doe/utils.py#L700) | Screens patterns against input bounds and intrapoint linear constraints. Returns immediately without patterns or linear constraints. | Filters the existing pattern generator's output without changing its enumeration rules. |
| [Constraint conversion](bofire/strategies/doe/utils.py#L717) | Reuses the existing constraint-to-matrix helper once per constraint. Separates equality and inequality rows. | Avoids duplicating coefficient ordering and normalization logic. |
| [Bounds and matrices](bofire/strategies/doe/utils.py#L730) | Builds local bounds and SciPy matrices. | Checks must not modify the caller's domain. SciPy's abbreviated names are explained beside the assignments. |
| [Interval checks and joint LP](bofire/strategies/doe/utils.py#L741) | Copies bounds per pattern, pins inactive coordinates to zero, and computes each linear expression's attainable minimum and maximum. Calls `linprog` only when multiple individually feasible rows remain. | A single linear expression over a box attains every value between its minimum and maximum, so one row needs no LP. Multiple rows may each be attainable but mutually contradictory. |
| [Builder integration](bofire/strategies/doe/utils.py#L786) | Filters after existing merge/deduplication. Raises a distinct error if linear feasibility eliminates every generated pattern. | Each unique pattern is checked once, before experiment assignment. Existing contradictory-overlap handling remains separate. |
| [Bounds-assignment documentation](bofire/strategies/doe/utils.py#L821) | Documents screening, reuse, and domain immutability. | A pattern is not a candidate. Existing cycling can assign six feasible patterns to more than six experiments. |

**Error handling:** LP success retains the pattern; status 2 drops it as infeasible.
Other solver failures raise rather than silently deleting search space.
The interval tolerance is `1e-7`, matching HiGHS' default primal feasibility tolerance.

**Review boundaries:**

- This establishes linear feasibility per pattern, not feasibility under nonlinear,
  interpoint, or additional fixed-candidate restrictions.
- An unpinned variable may still optimize to zero. This does not establish minimum
  active counts or distinct experiments.
- Existing pattern enumeration, including `none_also_valid` behavior, is not redesigned.
- Explicitly all-zero coefficient vectors remain outside this fix, as agreed.
  Their special-case branch and dedicated test were removed.

## 2. RandomStrategy: validate the reduced polytope

**File:** [bofire/strategies/random.py](bofire/strategies/random.py#L40)

### New fixed-constraint helper

[Changed helper](bofire/strategies/random.py#L40):

1. Identify fixed versus free terms.
2. Subtract fixed contributions from the right-hand side.
3. Retain nonzero free terms with their original feature indices.
4. If no terms remain, check the constant statement and discard it only if satisfied.

For example, `2*a+b=5` with `a=1` becomes `b=3`. If both variables are fixed,
the result may instead be a satisfied or impossible constant statement.
BoTorch inequalities use `>=`, so their constant check is `0 >= rhs`, not `<=`.

**Defense:** a legitimate nonconstant constraint can become constant after NChooseK
pinning. This is necessary handling for #842, not support added for explicitly
all-zero input vectors. Violations raise `InfeasibilityError`, which the subset
sampling caller can recognize and recover from.

### Changes inside the polytope sampler

[Changed function](bofire/strategies/random.py#L422):

- Replaces the base version's one-pass singleton-equality extraction with repeated reduction.
  Fixing one coordinate can make another equality determine another coordinate.
- Checks each equality-implied value against the feature's bounds.
- Rejects equalities assigning conflicting values to the same coordinate.
- Allows machine-scale arithmetic roundoff at a boundary and clamps that tiny
  discrepancy to the actual bound; genuinely out-of-bounds values are rejected.
- Substitutes the resulting fixed values into inequalities too.
- [Moves constraint preparation](bofire/strategies/random.py#L536)
  before the all-fixed return path. Constant linear contradictions cannot bypass
  checks simply because there are no coordinates left to sample.

**Defense:** for `a+b=1`, pinning `b=0` requires `a=1`. If `a` is capped at `0.6`,
that subset is infeasible; returning `a=1` is not acceptable. The hit-and-run
sampling algorithm itself is unchanged.

The [normalization helper](bofire/strategies/random.py#L76) scales each row using
the norm of its original coefficients, including those removed when features were
fixed. This keeps equivalent constraints consistent for both constant checks and
the sampler, even when their coefficients are very small.

**Review points:** the constant/conflict checks use `1e-8`; boundary clamping uses
a magnitude-scaled machine-epsilon tolerance. These have different purposes and
are not a single unified tolerance policy. The chained-equality regression is
preservation coverage: the base version could already produce the expected result in that case.

## 3. RandomStrategy: recover from rejected subsets without losing rows

[Changed function](bofire/strategies/random.py#L212):

| Situation | Base behavior | Replacement |
|---|---|---|
| Every subset in the draw is infeasible | Immediately raises. | [Retries subset batches](bofire/strategies/random.py#L256) up to `max_iters`, caching rejected subsets for this call. |
| Some subsets succeed, but too few rows remain | Final sampling can request more rows than available. | [Replenishes the shortfall](bofire/strategies/random.py#L302) with fresh samples from surviving subset domains. |
| The search budget is exhausted | Error implies only the sampled subsets failed, without recovery. | Error states the budget and rejected count, explains that this is not proof of domain infeasibility, and suggests settings to investigate. |

**Defense:**

- `ask(1)` can initially draw an impossible singleton even when feasible pairs exist.
  Immediate failure would turn a valid domain into a seed-dependent failure.
- Replenishment uses new sampler calls, not duplication of rows already obtained.
- Allocation follows surviving drawn multiplicities, with integer rounding.
  The 3:1 test supplies one feasible subset three times and another once: it checks
  preservation of those supplied weights, not a universal 3:1 distribution.
- The original domain is copied before pinning.

**Intentional limits:** retrying is bounded, not exhaustive. A feasible domain can
still exhaust the budget. The loop stops at the first batch with any success;
replenishment uses that batch's surviving subsets. This does not promise a uniform
distribution over all globally feasible subsets, nor global candidate uniqueness.
Rejected-subset caching relies on `InfeasibilityError` identifying an infeasible
reduced problem. Separately sampled groups can violate interpoint constraints
across the final batch. The [shared-setting check](bofire/strategies/random.py#L178)
raises a clear error if that happens, including after nonlinear rejection. It does
not generate a replacement batch that satisfies interpoint constraints; a full
batch-wide solution remains a separate issue.

`max_iters` is reused rather than adding a new setting. Subset retries and nonlinear
rejection have separate counters, so nesting can multiply work; it is not one shared
overall-call budget. `nchoosek_max_iters` remains the separate overlapping-NChooseK
subset-generation budget.

## 4. Preserve coefficients in solver precision

**File:** [bofire/utils/torch_tools.py](bofire/utils/torch_tools.py#L127)

Both conversion branches construct coefficient tensors with `**tkwargs` directly,
instead of constructing a tensor first and then calling `.to(**tkwargs)`.
The [source comment](bofire/utils/torch_tools.py#L135) explains the numerical reason.

**Defense:** the unscaled Python list previously became a default-float32 tensor
before conversion to float64. That rounds `0.1` to approximately `0.10000000149`.
With `b=0`, `0.1*a+b=0.3` then implies approximately `2.999999955`, below a valid
lower bound of 3. The stricter bounds validation exposed this pre-existing conversion
loss, making its correction tightly coupled to this patch.

The scaled NumPy path already retained precision; its edit makes construction
consistent rather than fixing the same demonstrated rounding there.
The `unit_scaled` flag and scaling algorithm existed in HEAD.

## 5. Tests: what each addition actually establishes

### DoE tests

**File:** [tests/bofire/strategies/doe/test_utils.py](tests/bofire/strategies/doe/test_utils.py#L43)

| Test link | Purpose / defense |
|---|---|
| [Mixture caps](tests/bofire/strategies/doe/test_utils.py#L54) | Checks empty/reduced feasible sets, pair and singleton feasibility boundaries, repeated calls, cycling, and domain immutability. Two max-count values and seven caps make 14 cases, not an unrestricted cross-product. |
| [Linear scenarios](tests/bofire/strategies/doe/test_utils.py#L96) | Covers inequality rejection, joint contradiction, no linear constraints, and no pinned features. The joint case rejects everything; it does not alone demonstrate selective joint rejection with other patterns surviving. |
| [Feature ordering](tests/bofire/strategies/doe/test_utils.py#L155) | Asymmetric bounds and reordered names/coefficients expose index mismatches, including patterns pinning multiple features. |
| [Signed coefficients](tests/bofire/strategies/doe/test_utils.py#L201) | Verifies negative terms contribute the correct attainable extremes for both equality and inequality rows. |
| [Skip LP](tests/bofire/strategies/doe/test_utils.py#L230) | Preservation/performance guard: no solver calls without patterns or without linear rows. Not an original candidate-generation failure. |
| [Deduplicate before LP](tests/bofire/strategies/doe/test_utils.py#L260) | New implementation contract: two unique patterns incur two LP calls, even when assigned to many experiments. |
| [Surface solver failure](tests/bofire/strategies/doe/test_utils.py#L293) | New failure contract: numerical failure must raise rather than masquerade as infeasibility. |
| [Actual strategy.ask](tests/bofire/strategies/doe/test_utils.py#L322) | End-to-end requests for 5, 6, and 8 rows exercise below/equal/above the six feasible pair patterns. This checks candidate output, not merely an internal list. |

### RandomStrategy tests

**File:** [tests/bofire/strategies/test_random.py](tests/bofire/strategies/test_random.py#L31)

| Test link | Purpose / defense |
|---|---|
| [Pinned mixture](tests/bofire/strategies/test_random.py#L35) | Tests constant contradiction with every component pinned and an out-of-bounds implied value with one free capped component. |
| [Equivalent scaled constraints](tests/bofire/strategies/test_random.py#L57) | Checks that equality and inequality constraints behave identically at ordinary and small coefficient scales, including retrying after an impossible subset. |
| [Replenishment](tests/bofire/strategies/test_random.py#L98) | Deterministic supplied subset draws plus real polytope sampling check exact row counts, validity, immutability, and surviving multiplicities. |
| [Shared-setting guard](tests/bofire/strategies/test_random.py#L141) | Checks clear failure for whole-batch and grouped constraints, including with validation disabled; [valid batches](tests/bofire/strategies/test_random.py#L174) and [valid replenishment](tests/bofire/strategies/test_random.py#L201) remain accepted, and [nonlinear rejection](tests/bofire/strategies/test_random.py#L232) is checked after combining proposals. |
| [Fixed conflicts](tests/bofire/strategies/test_random.py#L268) | Separates bound, equality, inequality, and zero-after-substitution contradictions. The latter starts with nonzero coefficients; it is not an explicitly all-zero input row. |
| [Chained equalities](tests/bofire/strategies/test_random.py#L311) | Preservation check for valid implied values through multiple equalities. |
| [Decimal boundary](tests/bofire/strategies/test_random.py#L338) | Protects actual returned candidates against the rounding regression exposed by stricter bounds checks. |
| [Retry/rejection cache](tests/bofire/strategies/test_random.py#L366) | Controlled draws prove recovery on the final allowed batch, budget exhaustion, and no repeated polytope work for cached rejected subsets. |
| [One-candidate requests](tests/bofire/strategies/test_random.py#L427) | Three fixed seeds exercise actual subset drawing and `ask(1)`. These complement, rather than replace, deterministic retry tests. |

### Conversion test

**File:** [tests/bofire/utils/test_torch_tools.py](tests/bofire/utils/test_torch_tools.py#L454)

Two cases check scaled/unscaled coefficients exactly. Exact equality is deliberate:
a loose tolerance can conceal float32 rounding. Scaling by two does not introduce
inexact decimal arithmetic beyond the original coefficient representation.
The unscaled case catches the conversion bug; the scaled case is preservation
coverage. This overlaps the end-to-end decimal test but localizes the cause.

## 6. Opt-in performance comparison

**New file:** [tests/bofire/strategies/doe/test_utils_performance.py](tests/bofire/strategies/doe/test_utils_performance.py#L1)

- [Reference measurements](tests/bofire/strategies/doe/test_utils_performance.py#L38)
  compare old-style generation/deduplication against filtering. All times are milliseconds.
- [Fifteen cases](tests/bofire/strategies/doe/test_utils_performance.py#L51):
  five scenarios at 4, 8, and 12 features, avoiding large pattern explosions.
- [LP instrumentation](tests/bofire/strategies/doe/test_utils_performance.py#L100)
  checks pattern counts and solver-call counts separately from timing.
- [Timing loop](tests/bofire/strategies/doe/test_utils_performance.py#L115)
  warms both paths, alternates their order, and reports seven-run medians.

**Defense:** this answers the requested LP-cost comparison without a benchmark
dependency or flaky time-based assertions. It runs only when explicitly enabled.
Reference single-mixture filtering was about 1.1/3.6/6.0 ms with zero LPs;
mixed constraints were about 24/52/92 ms with 5/13/21 LPs.

**Limitation:** timing observations are not performance guarantees. A slowdown does
not fail this test; deterministic output and LP-call contracts do.

## 7. Remaining documentation files

| File and changed link | Summary and defense |
|---|---|
| [bofire/data_models/strategies/random.py:41](bofire/data_models/strategies/random.py#L41) | Documents subset rejection, replenishment, bounded retries, and the separate uses of `max_iters` and `nchoosek_max_iters`. No fields/defaults or serialization shape are added. |
| [docs/tutorials/doe/nchoosek_constraint.qmd:14](docs/tutorials/doe/nchoosek_constraint.qmd#L14) | Replaces the old warning about infeasible assignments with the filtering behavior, cost strategy, pattern reuse, and explicit limits. Removes the obsolete blanket prohibition on overlapping NChooseK constraints; merging itself already existed. |
| [CHANGELOG.md:10](CHANGELOG.md#L10) and [fix entry:28](CHANGELOG.md#L28) | Records the opt-in measurement command and functional fix, including retry/replenishment behavior. |
| [CLAUDE.md:112](CLAUDE.md#L112) | Records the explicitly requested pytest `monkeypatch` convention instead of `unittest.mock`. Contributor guidance only, not part of the runtime fix. |

## Validation evidence and caveats

Combined focused run after adding the shared-setting guard:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\bofire\strategies\doe\test_utils.py tests\bofire\strategies\test_random.py tests\bofire\utils\test_torch_tools.py tests\bofire\strategies\doe\test_utils_performance.py -k "not test_n_zero_eigvals_unconstrained and not test_n_zero_eigvals_constrained" -q
```

**Result:** 141 passed, 16 skipped, two known environment-dependent failures
deselected. The opt-in performance suite separately passed all 15 cases.

The following earlier combined run predates the scaled-constraint fix:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\bofire\strategies\doe\test_utils.py tests\bofire\strategies\test_random.py tests\bofire\utils\test_torch_tools.py -k "nchoosek or pseudo_fixed or fixed_conflict or get_linear_constraints" -q
```

**Result:** 52 passed, 83 deselected. The targeted DoE source/test Ruff lint and
format checks also passed at that point. A separate decimal-precision run passed
both cases. These are focused results, not a full-suite pass.

The opt-in performance cases were run earlier to obtain the recorded measurements;
they were not rerun merely to write this guide.

Earlier broader validation identified two existing `n_zero_eigvals` failures caused
by the pandas/formulaic type-registration mismatch. That problem was independently
reproduced with the original implementation; it is not fixed here.

## Questions to keep in view during review

- Are the feasibility scope and tolerance choices appropriate for the intended domains?
- Is the bounded, first-successful-batch retry policy acceptable, including its
  possible sampling bias and cost when nested inside nonlinear rejection?
- Should selective joint-LP rejection receive an additional regression beyond the
  current all-pattern contradiction case?
- Is the overlap between helper-level precision coverage and the end-to-end decimal
  test useful enough to retain?
- How should RandomStrategy preserve shared settings across separately sampled
  groups when an interpoint constraint applies to the complete batch?
