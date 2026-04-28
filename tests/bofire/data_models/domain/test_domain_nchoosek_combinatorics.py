import unittest

import numpy as np
import pandas as pd
import pytest

from bofire.data_models.constraints.api import NChooseKConstraint
from bofire.data_models.domain.api import Constraints, Domain, Inputs
from bofire.data_models.features.api import (
    CategoricalDescriptorInput,
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
)


# NChooseKConstraint constraints 1
cc1a = NChooseKConstraint(
    features=["0", "1", "2", "3"],
    min_count=2,
    max_count=3,
    none_also_valid=True,
)
cc2a = NChooseKConstraint(
    features=["2", "3", "4", "5"],
    min_count=1,
    max_count=2,
    none_also_valid=True,
)

# NChooseKConstraint constraints 2
cc1b = NChooseKConstraint(
    features=["0", "1", "2", "3"],
    min_count=2,
    max_count=3,
    none_also_valid=False,
)
cc2b = NChooseKConstraint(
    features=["2", "3", "4", "5"],
    min_count=1,
    max_count=2,
    none_also_valid=True,
)

# NChooseKConstraint constraint 3
cc3 = [
    NChooseKConstraint(
        features=["0", "1", "2", "3"],
        min_count=2,
        max_count=3,
        none_also_valid=True,
    ),
]

# input features
continuous_inputs = []
for i in range(6):
    f = ContinuousInput(key=str(i), bounds=(0, 1))
    continuous_inputs.append(f)
categorical_feature = CategoricalInput(
    key="categorical_feature",
    categories=["c1", "c2"],
)
categorical_descriptor_feature = CategoricalDescriptorInput(
    key="categorical_descriptor_feature",
    categories=["cd1", "cd2"],
    descriptors=["d1", "d2"],
    values=[[1.0, 1.0], [2.0, 2.0]],
)

# output feature
outputs = [ContinuousOutput(key="y")]


"""
TEST CASES:

CASE 1: 6 continuous features, 2 overlapping NChooseKConstraint constraints, none_also_valid: True, True
CASE 2: 6 continuous features, 2 overlapping NChooseKConstraint constraints, none_also_valid: False, True
"""

# CASE 1
test_features_used_1 = [
    ["0", "1", "2"],
    ["0", "1", "3"],
    ["0", "1", "4"],
    ["0", "1", "5"],
    ["0", "1", "2", "4"],
    ["0", "1", "2", "5"],
    ["0", "1", "3", "4"],
    ["0", "1", "3", "5"],
    ["0", "1", "4", "5"],
    ["0", "1"],
    ["0", "2"],
    ["0", "2", "3"],
    ["0", "2", "4"],
    ["0", "2", "5"],
    ["0", "3"],
    ["0", "3", "4"],
    ["0", "3", "5"],
    ["1", "2"],
    ["1", "2", "3"],
    ["1", "2", "4"],
    ["1", "2", "5"],
    ["1", "3"],
    ["1", "3", "4"],
    ["1", "3", "5"],
    ["2", "3"],
    ["4"],
    ["5"],
    ["4", "5"],
    [],
]
test_features_unused_1 = []
for used in test_features_used_1:
    unused = [f for f in ["0", "1", "2", "3", "4", "5"] if f not in used]
    test_features_unused_1.append(unused)

# CASE 2
test_features_used_2 = [
    ["0", "1", "2"],
    ["0", "1", "3"],
    ["0", "1", "4"],
    ["0", "1", "5"],
    ["0", "1", "2", "4"],
    ["0", "1", "2", "5"],
    ["0", "1", "3", "4"],
    ["0", "1", "3", "5"],
    ["0", "1", "4", "5"],
    ["0", "1"],
    ["0", "2"],
    ["0", "2", "3"],
    ["0", "2", "4"],
    ["0", "2", "5"],
    ["0", "3"],
    ["0", "3", "4"],
    ["0", "3", "5"],
    ["1", "2"],
    ["1", "2", "3"],
    ["1", "2", "4"],
    ["1", "2", "5"],
    ["1", "3"],
    ["1", "3", "4"],
    ["1", "3", "5"],
    ["2", "3"],
]
test_features_unused_2 = []
for used in test_features_used_2:
    unused = [f for f in ["0", "1", "2", "3", "4", "5"] if f not in used]
    test_features_unused_2.append(unused)

# experiments
experiments = pd.DataFrame(
    np.random.uniform(size=(24, 7)),
    columns=["0", "1", "2", "3", "4", "5", "y"],
)
experiments["categorical_feature"] = ["c1"] * 12 + ["c2"] * 12
experiments["categorical_descriptor_feature"] = (["cd1"] * 6 + ["cd2"] * 6) * 2
experiments["valid_y"] = 1


##### LIST OF TASTE CASES #####

test_cases = []

# CASE 1
test_case = {}
domain = Domain.from_lists(
    inputs=continuous_inputs,
    outputs=outputs,
    constraints=[cc1a, cc2a],
)
test_case["domain"] = domain
test_case["experiments"] = experiments
test_case["descriptor_method"] = None
test_case["categorical_method"] = None
test_case["descriptor_encoding"] = None
test_case["categorical_encoding"] = None
test_case["test_features_used"] = test_features_used_1
test_case["test_features_unused"] = test_features_unused_1
test_cases.append(test_case)

# CASE 2
test_case = {}
domain = Domain.from_lists(
    continuous_inputs,
    outputs,
    constraints=[cc1b, cc2b],
)
test_case["domain"] = domain
test_case["experiments"] = experiments
test_case["descriptor_method"] = None
test_case["categorical_method"] = None
test_case["descriptor_encoding"] = None
test_case["categorical_encoding"] = None
test_case["test_features_used"] = test_features_used_2
test_case["test_features_unused"] = test_features_unused_2
test_cases.append(test_case)


@pytest.mark.parametrize("test_case", test_cases)
def test_nchoosek_combinations_completeness(test_case):
    domain = test_case["domain"]
    features_used, features_unused = domain.get_nchoosek_combinations(exhaustive=True)
    for features in test_case["test_features_used"]:
        assert features in features_used
    for features in test_case["test_features_unused"]:
        assert features in features_unused


def test_nchoosek_combinations_nonexhaustive():
    domain = Domain(
        inputs=Inputs(
            features=[
                ContinuousInput(key=f"if{i + 1}", bounds=(0, 1)) for i in range(6)
            ],
        ),
        constraints=Constraints(
            constraints=[
                NChooseKConstraint(
                    features=[f"if{i + 1}" for i in range(4)],
                    min_count=0,
                    max_count=2,
                    none_also_valid=True,
                ),
            ],
        ),
    )
    used, unused = domain.get_nchoosek_combinations(exhaustive=False)
    expected_used = [
        ["if1", "if2"],
        ["if1", "if3"],
        ["if1", "if4"],
        ["if2", "if3"],
        ["if2", "if4"],
        ["if3", "if4"],
    ]
    expected_unused = [
        ["if3", "if4"],
        ["if2", "if4"],
        ["if2", "if3"],
        ["if1", "if4"],
        ["if1", "if3"],
        ["if1", "if2"],
    ]
    # print(combos, expected_combos)
    c = unittest.TestCase()
    c.assertCountEqual(used, expected_used)
    c.assertCountEqual(unused, expected_unused)


def test_sample_valid_nchoosek_features_uniform_over_subsets():
    """With one NChooseK on n=5 features and k in [1, 3], there are
    C(5,1)+C(5,2)+C(5,3) = 25 valid subsets. With uniform sampling each
    should appear with frequency ~1/25.
    """
    import random

    inputs = [ContinuousInput(key=f"x{i}", bounds=(0, 1)) for i in range(5)]
    constraint = NChooseKConstraint(
        features=[f"x{i}" for i in range(5)],
        min_count=1,
        max_count=3,
        none_also_valid=False,
    )
    domain = Domain(
        inputs=Inputs(features=inputs),
        constraints=Constraints(constraints=[constraint]),
    )
    n_samples = 25_000
    samples = domain.sample_valid_nchoosek_features(random.Random(0), n=n_samples)
    counts: dict = {}
    for s in samples:
        counts[s] = counts.get(s, 0) + 1
    assert len(counts) == 25, f"Expected 25 unique subsets, got {len(counts)}"
    expected = n_samples / 25
    for subset, count in counts.items():
        rel = abs(count - expected) / expected
        assert (
            rel < 0.20
        ), f"Subset {subset} count {count} too far from expected {expected:.0f}"


def test_sample_valid_nchoosek_features_none_also_valid():
    """When none_also_valid=True, the empty subset is in the support."""
    import random

    inputs = [ContinuousInput(key=f"x{i}", bounds=(0, 1)) for i in range(3)]
    constraint = NChooseKConstraint(
        features=["x0", "x1", "x2"],
        min_count=2,
        max_count=3,
        none_also_valid=True,
    )
    domain = Domain(
        inputs=Inputs(features=inputs),
        constraints=Constraints(constraints=[constraint]),
    )
    samples = domain.sample_valid_nchoosek_features(random.Random(1), n=2000)
    unique = set(samples)
    # Valid subsets: () + C(3,2) + C(3,3) = 1 + 3 + 1 = 5
    assert len(unique) == 5
    assert () in unique


def test_sample_valid_nchoosek_features_allow_zero_singletons():
    """Without any NChooseK, allow_zero=True features form singleton groups."""
    import random

    inputs = [
        ContinuousInput(key="a", bounds=(0.1, 1), allow_zero=True),
        ContinuousInput(key="b", bounds=(0.1, 1), allow_zero=True),
        ContinuousInput(key="c", bounds=(0.1, 1)),
    ]
    domain = Domain(inputs=Inputs(features=inputs))
    samples = domain.sample_valid_nchoosek_features(random.Random(2), n=2000)
    unique = set(samples)
    # Each of {a, b} can be on or off independently -> 4 subsets
    assert unique == {(), ("a",), ("b",), ("a", "b")}


def test_sample_valid_nchoosek_features_empty_returns_empty_tuple():
    """Domain without NChooseK and without allow_zero features yields ()."""
    import random

    inputs = [ContinuousInput(key="x", bounds=(0, 1))]
    domain = Domain(inputs=Inputs(features=inputs))
    samples = domain.sample_valid_nchoosek_features(random.Random(3), n=4)
    assert samples == [(), (), (), ()]


def test_sample_valid_nchoosek_features_default_n_is_one():
    """Default returns a list of length 1."""
    import random

    inputs = [ContinuousInput(key=f"x{i}", bounds=(0, 1)) for i in range(3)]
    constraint = NChooseKConstraint(
        features=["x0", "x1", "x2"],
        min_count=1,
        max_count=2,
        none_also_valid=False,
    )
    domain = Domain(
        inputs=Inputs(features=inputs),
        constraints=Constraints(constraints=[constraint]),
    )
    samples = domain.sample_valid_nchoosek_features(random.Random(0))
    assert len(samples) == 1
