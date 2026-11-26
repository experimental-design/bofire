"""Opt-in measurement of NChooseK linear-filter overhead, not optimizer speed.

Run in PowerShell:
    $env:BOFIRE_RUN_PERFORMANCE_TESTS = '1'
    python -m pytest -q -s tests\\bofire\\strategies\\doe\\test_utils_performance.py

Ordinary runs skip this comparison. Timings are informational: machine load,
SciPy version, and hardware affect them, so CI asserts correctness and LP-call
counts rather than an arbitrary wall-clock threshold. No benchmark plugin is needed.
The single-row cases intentionally use no LP: their attainable interval suffices.
The mixed case exercises actual joint LPs after cheap impossible-pattern rejection.
"""

import os
import platform
from math import comb
from statistics import median
from time import perf_counter

import pytest
import scipy

from bofire.data_models.constraints.api import (
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
)
from bofire.data_models.domain.api import Domain
from bofire.data_models.features.api import ContinuousInput
from bofire.strategies.doe import utils


pytestmark = pytest.mark.skipif(
    os.environ.get("BOFIRE_RUN_PERFORMANCE_TESTS") != "1",
    reason="set BOFIRE_RUN_PERFORMANCE_TESTS=1 to measure pattern-filter overhead",
)

# Reference run: Windows AMD64, Python 3.14.10, SciPy 1.18.1; seven warmed
# repetitions per case. Columns below are baseline -> filtered median ms
# (LP calls), for 4 / 8 / 12 features respectively:
# NChooseK only: .080 -> .103 (0), .211 -> .240 (0), .604 -> .612 (0)
# Linear only:   .003 -> .033 (0), .002 -> .020 (0), .003 -> .030 (0)
# Mixture:       .132 -> 1.115 (0), .370 -> 3.558 (0), .755 -> 5.957 (0)
# Feasible:      .089 -> .890 (0), .327 -> 2.962 (0), .705 -> 5.178 (0)
# Mixed:         .273 -> 24.214 (5), .362 -> 51.501 (13), .739 -> 91.782 (21)
# These are observations, not budgets or guarantees; even repeated runs vary.
# Initial LP-only filtering took ~32 / 84 / 190 ms for the mixture cases,
# motivating the exact single-row interval path measured here.


@pytest.mark.parametrize("size", [4, 8, 12])
@pytest.mark.parametrize(
    "variant", ["nchoosek_only", "linear_only", "mixture", "feasible", "mixed"]
)
def test_pattern_filter_performance(size, variant, monkeypatch):
    """Issue #842: measure filtering overhead against unfiltered pattern generation."""
    # Keep domain construction outside timing. max_count=2 gives 10, 36, and
    # 78 patterns: enough to expose per-LP cost without exponential test growth.
    keys = [f"x{i:02}" for i in range(size)]
    constraints = []
    if variant != "linear_only":
        constraints.append(
            NChooseKConstraint(
                features=keys, min_count=1, max_count=2, none_also_valid=False
            )
        )
    if variant != "nchoosek_only":
        constraints.append(
            LinearEqualityConstraint(features=keys, coefficients=[1] * size, rhs=1)
        )
    if variant == "mixed":
        constraints.append(
            LinearInequalityConstraint(
                features=keys[:2], coefficients=[-1, -1], rhs=-0.1
            )
        )
    domain = Domain.from_lists(
        inputs=[
            ContinuousInput(
                key=key, bounds=(0, 0.6 if variant in ("mixture", "mixed") else 1)
            )
            for key in keys
        ],
        constraints=constraints,
    )

    def baseline():
        # Reproduce the old builder's generation/deduplication, not just enumeration.
        return list(set(utils._get_nchoosek_combined_patterns(domain)))

    expected_input = 0 if variant == "linear_only" else size + comb(size, 2)
    expected_output = {
        "linear_only": 0,
        "nchoosek_only": expected_input,
        "feasible": expected_input,
        "mixture": comb(size, 2),
        "mixed": comb(size, 2) - comb(size - 2, 2),
    }[variant]
    assert len(baseline()) == expected_input

    # Instrument separately: the counting wrapper must not contaminate timings.
    linprog = utils.opt.linprog
    calls = 0

    def counted_lp(*args, **kwargs):
        nonlocal calls
        calls += 1
        return linprog(*args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(utils.opt, "linprog", counted_lp)
        assert len(utils._build_nchoosek_combined_patterns(domain)) == expected_output
    assert calls == (expected_output if variant == "mixed" else 0)

    # Warm both paths (including solver initialization), then alternate measurement
    # order across seven paired repetitions to reduce ordering/load bias.
    baseline()
    utils._build_nchoosek_combined_patterns(domain)
    times = {"baseline": [], "filtered": []}
    for repetition in range(7):
        functions = [
            ("baseline", baseline),
            ("filtered", lambda: utils._build_nchoosek_combined_patterns(domain)),
        ]
        if repetition % 2:
            functions.reverse()
        for label, function in functions:
            start = perf_counter()
            result = function()
            times[label].append(perf_counter() - start)
            assert len(result) == (
                expected_input if label == "baseline" else expected_output
            )
    old_ms = median(times["baseline"]) * 1000
    new_ms = median(times["filtered"]) * 1000
    print(
        f"\nPython {platform.python_version()}, SciPy {scipy.__version__}, "
        f"{platform.system()} {platform.machine()}: "
        f"{variant}, n={size}, patterns={expected_input}->{expected_output}, "
        f"LP calls={calls}, median baseline={old_ms:.3f}ms, "
        f"filtered={new_ms:.3f}ms, overhead={new_ms - old_ms:.3f}ms, "
        f"ratio={new_ms / old_ms:.2f}x"
    )
