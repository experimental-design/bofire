# Greedy Pruning for NChooseK Constraints

This document describes the greedy pruning post-processing step used to enforce
NChooseK (cardinality) constraints during acquisition function optimization. The
algorithm is a direct adaptation of the BONSAI procedure
(Daulton et al., *Bayesian Optimization with Natural Simplicity and
Interpretability*, arXiv:2602.07144) to BoFire's setting of mixed continuous /
semi-continuous features and (optionally) overlapping linear constraints.

## Motivation

NChooseK constraints are non-convex: they require that, out of a set of
candidate features, at most `max_count` (and at least `min_count`) take a
strictly positive value. Encoding this directly in the acquisition function
optimizer turns the problem into a mixed-integer program, which is expensive,
brittle, and forces the optimizer onto disconnected feasible regions.

The BONSAI insight is that we don't need to encode the cardinality constraint
inside the optimizer at all. Instead:

1. Optimize the acquisition function on a *convex relaxation* of the feasible
   set, ignoring the cardinality constraint.
2. Post-process the resulting candidate by greedily pushing features off the
   active set until the cardinality constraint is satisfied, choosing at each
   step the feature whose removal costs the least acquisition value.

This is the same pattern the original BONSAI paper applies to "deviations from a
default configuration", with the default value here taken to be zero.

## Setting

Each feature `x_j` has

- a lower bound `lb_j ≥ 0` and upper bound `ub_j > lb_j`,
- optionally an `allow_zero` flag, indicating that the value `0` is also
  feasible even when `lb_j > 0` (i.e. the feasible set per feature is the
  disconnected union `{0} ∪ [lb_j, ub_j]`).

A feature is called **semi-continuous** when `allow_zero` is true and
`lb_j > 0`. Note that a single semi-continuous feature is structurally a
one-feature NChooseK constraint with `min_count = 0`, `max_count = 1`,
`none_also_valid = True`. The algorithm treats both cases uniformly.

The feasible set may also include linear equality and inequality constraints
that *overlap* the NChooseK feature set (e.g. a mixture constraint
`Σ x_j = 1` over the same features that participate in the NChooseK).
Nonlinear or interpoint constraints overlapping the NChooseK feature set are
**not supported** by this algorithm — they are blocked at the domain level and
fall back to the standard nonlinear-constrained acquisition optimization.

## Algorithm

### Step 0 — Acquisition function optimization on the convex relaxation

The acquisition function is optimized over the box `[0, ub_j]` for every
feature, with the linear constraints kept active. The semi-continuity gap
`(0, lb_j)` and the NChooseK cardinality constraint are *both* relaxed away at
this stage. The optimizer therefore sees a convex (or at least well-behaved)
feasible set and can use gradients freely.

The result is a dense candidate `x*` in which every feature falls into one of
three states:

- **zero:** `x*_j = 0`. No further decision needed.
- **fractional:** `x*_j ∈ (0, lb_j)`. Violates semi-continuity. Must be either
  zeroed or snapped into `[lb_j, ub_j]`.
- **cleanly active:** `x*_j ∈ [lb_j, ub_j]`. No semi-continuity issue, but
  may still need to be zeroed to satisfy NChooseK.

Let `a` denote the active count `|{j : x*_j > 0}|`.

### Step 1 — Greedy pruning loop

The loop maintains a current candidate that starts at `x*` and, at each
iteration, commits one *action* that moves a single feature into a terminal
state (`0` or `[lb_j, ub_j]`). Two kinds of actions are considered.

**Zero action `zero(j)`** — pin `x_j = 0` and re-establish feasibility of the
remaining features against the linear constraints. If a feature `j` is currently
active (fractional or cleanly active), this is always available. Committing
`zero(j)` decrements the active count by one.

**Active action `active(j)`** — only available for currently fractional
features. Set the bounds of `j` to `[lb_j, ub_j]` and re-establish feasibility.
Committing `active(j)` resolves the semi-continuity violation for `j` without
changing the active count.

For each candidate action, a per-action *variant* candidate is constructed (see
hyperparameters below). The acquisition function is evaluated at each variant.
The action with the **highest acquisition value** (equivalently, the smallest
acquisition reduction relative to the dense AF value) is committed.

### Step 2 — Termination and the min_count guard

The loop terminates when *both* of the following hold:

- no feature is fractional, and
- `min_count ≤ a ≤ max_count`.

The upper bound `a ≤ max_count` is enforced by exhaustion: once all fractional
features have been resolved, the action set contains only zero variants, and
each commit decrements `a` by one until the bound is met.

The lower bound `a ≥ min_count` is **not** automatically respected by the
greedy rule, because the rule is budget-blind: it always picks the action with
smallest acquisition loss, even if that action would drop `a` below `min_count`.
This is what the **min_count guard** prevents.

The guard is a filter applied to the action set at every iteration: a
`zero(j)` action is removed from the set whenever committing it would bring
`a − 1 < min_count`. (When `none_also_valid = True` the guard also tolerates
`a = 0` as a special case.) Active actions are never filtered. If the filtered
action set is empty before the constraint is met — for example, because every
remaining variant is QP-infeasible — the algorithm raises rather than returning
an infeasible candidate.

### Properties

- Each iteration commits exactly one action, and each fractional feature can
  appear in at most one active action. The loop therefore terminates in at most
  `n` iterations, where `n` is the number of features in the NChooseK group.
- Linear feasibility is enforced at every step by construction: each variant is
  built with `x_j` either fixed at zero or constrained to `[lb_j, ub_j]`, and
  projected onto the linear constraint set. Variants whose projection is
  infeasible are dropped from the action set for that iteration.
- The acquisition value is monotonically traded for feasibility in the BONSAI
  sense: at every step, the smallest available acquisition loss is incurred.
- The procedure is deterministic given a fixed acquisition function and a
  fixed dense candidate `x*`.

## Multiple NChooseK constraints

The algorithm extends to domains with several NChooseK constraints — possibly
on disjoint feature sets, possibly overlapping — without changing its
selection rule. Three book-keeping changes are needed.

### Per-constraint active counts

Instead of a single active count `a`, the loop tracks an active count `a_c`
for every NChooseK constraint `c`. A commit on a feature `j` updates `a_c`
for every constraint `c` whose feature set contains `j`. Disjoint
constraints update independently; overlapping constraints update in lock-step
on shared features.

### Termination is a conjunction over all constraints

The loop terminates when *no* feature is fractional and *every* NChooseK
constraint is satisfied:

```
fractional == ∅   AND   ∀ c :  min_count_c ≤ a_c ≤ max_count_c
```

If even one constraint is still violated, the loop continues.

### The min_count guard becomes per-constraint

A `zero(j)` action is filtered out of the action set whenever it would push
`a_c` below `min_count_c` for any constraint `c` that contains `j` (with the
usual `none_also_valid` exception that allows `a_c = 0`). Active actions are
never filtered. As before, an empty filtered action set before all
`max_count_c` are satisfied indicates a mutually infeasible configuration:
the algorithm raises rather than returning an infeasible candidate.

### A useful efficiency restriction

When NChooseK constraints overlap with cleanly-active features, it is
worthwhile to further restrict the zero-action set to features that
participate in at least one *currently violated* constraint
(`a_c > max_count_c` for some `c ∋ j`). Zeroing a feature outside every
violated constraint costs acquisition value without contributing to
feasibility, and the greedy rule would never pick such an action anyway —
making the restriction a free pruning of the action set rather than a change
in behaviour.

### Termination bound

Each commit either resolves a fractional feature (one active or zero
commit per fractional feature) or zeros a cleanly-active feature whose
removal closes a `max_count_c` gap. The total number of iterations is
therefore bounded by

```
|fractional| + Σ_c  max(0, a_c − max_count_c)
```

evaluated at the dense relaxation candidate.

### Disjoint vs. overlapping feature sets

Disjoint NChooseK constraints are essentially independent: the loop iterates
over the union of their feature sets, but no commit on one constraint's
features ever affects another constraint's `a_c`. The interesting case is
overlapping constraints, where a single zero commit can simultaneously
reduce the violation of several constraints (the favourable interaction)
or simultaneously block several min_count guards (the failure mode that
forces the algorithm to raise).

## Hyperparameters

The algorithm exposes two knobs that trade computational cost against the
fidelity of the per-action acquisition value estimate.

### `per_step_local_reopt: bool`

Controls how each action's variant candidate is constructed.

- **`False` (project only).** When linear constraints overlap the NChooseK
  feature set, the dense candidate is QP-projected onto the linear feasible set
  with the per-action bound (`x_j = 0` for `zero`, `x_j ∈ [lb_j, ub_j]` for
  `active`) applied. The acquisition value is evaluated directly at this
  projected point. When no linear overlap exists, the projection collapses to
  simple coordinate replacement (zeroing `x_j` or clamping it into `[lb_j,
  ub_j]`).
- **`True` (project then locally reoptimize).** After QP projection, the
  acquisition function is locally re-optimized starting from the projection,
  with the per-action bound retained. The acquisition value of each action then
  reflects the *best achievable* AF value under that bound, rather than the
  acquisition at an arbitrary feasible warm-start.

Project-only is cheaper by a factor equal to the per-action local solver cost
and is usually good enough when the acquisition surface is smooth on the scale
of the linear-projection displacement. Local reoptimization is more accurate
and is recommended when the acquisition is sharp near the boundary of `lb_j`
(e.g. peaks immediately above `lb_j`).

### `final_local_reopt: bool`

Controls whether the final pruned candidate is locally reoptimized as a
clean-up step.

- **`False`.** The candidate returned by the greedy loop is returned as-is.
  Its components are guaranteed to lie in `{0} ∪ [lb_j, ub_j]` and to satisfy
  all linear and NChooseK constraints by construction.
- **`True`.** A single local acquisition optimization is run with the active
  set frozen — features that the loop committed to zero stay at zero, and
  features that the loop committed to active remain in `[lb_j, ub_j]`. This
  cleans up small drifts introduced by the per-step QP projections and lets the
  optimizer settle into the local AF maximum within the now-feasible region.

The cost of `final_local_reopt = True` is one extra local solve, independent of
the size of the NChooseK group, and it is usually worth enabling unless the
acquisition optimizer call is expensive.

The two knobs are independent. A useful default combination is
`per_step_local_reopt = False, final_local_reopt = True`: cheap per-action
estimates drive the greedy decisions, and a single end-of-pipeline solve
absorbs the accumulated rounding from the QP projections.

## Worked example

Consider a domain with four features `x1, x2, x3, x4`, each with bounds
`(0, 1)`, `allow_zero = True` and `lb = 0.2`, subject to

- a mixture equality `x1 + x2 + x3 + x4 = 1`,
- an NChooseK constraint on `{x1, x2, x3, x4}` with `min_count = 1`,
  `max_count = 2`, `none_also_valid = False`.

### Step 0

Acquisition function maximization on the convex relaxation
`x_j ∈ [0, 1], Σ x_j = 1` returns

```
x* = (0.60, 0.30, 0.05, 0.05)
```

State of features:

| feature | value | state              |
|---------|-------|--------------------|
| x1      | 0.60  | cleanly active     |
| x2      | 0.30  | cleanly active     |
| x3      | 0.05  | fractional         |
| x4      | 0.05  | fractional         |

Active count `a = 4`, exceeding `max_count = 2`.

### Iteration 1

The exit condition fails: `fractional = {x3, x4}` is non-empty and `a = 4`
exceeds the budget.

The action set has zero variants for every active feature and active variants
for the fractional ones. After QP projection (and, optionally, local
reoptimization) onto the mixture constraint, illustrative acquisition values
are:

| action      | candidate after projection       | AF value |
|-------------|----------------------------------|----------|
| zero(x1)    | (0.00, 0.60, 0.20, 0.20)         | 0.71     |
| zero(x2)    | (0.70, 0.00, 0.20, 0.10)         | 0.78     |
| zero(x3)    | (0.62, 0.32, 0.00, 0.06)         | 0.84     |
| zero(x4)    | (0.62, 0.32, 0.06, 0.00)         | 0.84     |
| active(x3)  | (0.55, 0.25, 0.20, 0.00)         | 0.82     |
| active(x4)  | (0.55, 0.25, 0.00, 0.20)         | 0.82     |

The min_count guard would block any action that brings `a` below 1; here every
zero action leaves `a ≥ 3` so none are filtered.

The greedy rule picks the action with the highest acquisition value. Both
`zero(x3)` and `zero(x4)` are tied at `0.84`. Break the tie deterministically
(say, in feature order) and commit `zero(x3)`.

```
candidate ← (0.62, 0.32, 0.00, 0.06)        a = 3,   fractional = {x4}
```

Notice that `zero(x3)` resolved both the semi-continuity violation on `x3`
*and* freed one NChooseK slot in a single move — the BONSAI rule arbitrates
between the two purposes naturally.

### Iteration 2

The exit condition still fails: `fractional = {x4}` and `a = 3 > 2`.

Re-evaluate the action set against the new candidate. `x3` is now permanently
zero and is removed from consideration.

| action      | candidate after projection       | AF value |
|-------------|----------------------------------|----------|
| zero(x1)    | (0.00, 0.70, 0.00, 0.30)         | 0.69     |
| zero(x2)    | (0.80, 0.00, 0.00, 0.20)         | 0.75     |
| zero(x4)    | (0.66, 0.34, 0.00, 0.00)         | 0.83     |
| active(x4)  | (0.55, 0.25, 0.00, 0.20)         | 0.82     |

The greedy rule picks `zero(x4)` at `0.83` and commits.

```
candidate ← (0.66, 0.34, 0.00, 0.00)        a = 2,   fractional = ∅
```

### Iteration 3

The exit condition holds: no feature is fractional and `a = 2` lies in
`[min_count, max_count] = [1, 2]`.

If `final_local_reopt` is enabled, a single local acquisition optimization is
run with `x3 = x4 = 0` fixed and `x1, x2 ∈ [0.2, 1]`. This may slightly refine
the values of `x1` and `x2` while preserving every constraint. Otherwise the
candidate is returned as is:

```
final candidate = (0.66, 0.34, 0.00, 0.00)
```

The candidate satisfies the mixture equality (`Σ = 1.0`), the per-feature
semi-continuity (`x ∈ {0} ∪ [0.2, 1]` for every coordinate), and the NChooseK
constraint (`a = 2 ≤ max_count = 2` and `a = 2 ≥ min_count = 1`).

## Comparison to the published BONSAI algorithm

The algorithm above differs from the published BONSAI procedure in two
respects:

1. **Default value.** BONSAI prunes deviations from an arbitrary user-supplied
   default configuration; here, the default is fixed at zero, which is the
   natural reference value for cardinality constraints.
2. **Termination.** BONSAI terminates when the relative acquisition loss
   exceeds a user-specified threshold `ρ`. The variant described here
   terminates when the NChooseK and semi-continuity constraints are exactly
   satisfied, with no AF-loss budget. This makes the procedure feasibility-first
   rather than acquisition-loss-first, which is the appropriate trade-off when
   the cardinality constraint is a hard requirement of the experimental design.
   The min_count guard is the corresponding feasibility-first analogue of
   BONSAI's "do not prune below the floor" rule.

The greedy selection rule itself — at each step, prefer the action with
smallest acquisition reduction — is unchanged.
