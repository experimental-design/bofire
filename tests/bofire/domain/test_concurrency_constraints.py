import numpy as np
import pandas as pd
import pytest

from bofire.domain.constraint import NChooseKConstraint
from bofire.domain.domain import Domain
from bofire.domain.feature import (
    CategoricalDescriptorInput,
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
)

# NChooseKConstraint constraints 1
cc1a = NChooseKConstraint(
    features=["0", "1", "2", "3"], min_count=2, max_count=3, none_also_valid=True
)
cc2a = NChooseKConstraint(
    features=["2", "3", "4", "5"], min_count=1, max_count=2, none_also_valid=True
)

# NChooseKConstraint constraints 2
cc1b = NChooseKConstraint(
    features=["0", "1", "2", "3"], min_count=2, max_count=3, none_also_valid=False
)
cc2b = NChooseKConstraint(
    features=["2", "3", "4", "5"], min_count=1, max_count=2, none_also_valid=True
)

# NChooseKConstraint constraint 3
cc3 = [
    NChooseKConstraint(
        features=["0", "1", "2", "3"], min_count=2, max_count=3, none_also_valid=True
    )
]

# input features
continuous_input_features = []
for i in range(6):
    f = ContinuousInput(key=str(i), lower_bound=0, upper_bound=1)
    continuous_input_features.append(f)
categorical_feature = CategoricalInput(
    key="categorical_feature", categories=["c1", "c2"]
)
categorical_descriptor_feature = CategoricalDescriptorInput(
    key="categorical_descriptor_feature",
    categories=["cd1", "cd2"],
    descriptors=["d1", "d2"],
    values=[[1.0, 1.0], [2.0, 2.0]],
)

# output feature
output_features = [ContinuousOutput(key="y")]


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
    np.random.uniform(size=(24, 7)), columns=["0", "1", "2", "3", "4", "5", "y"]
)
experiments["categorical_feature"] = ["c1"] * 12 + ["c2"] * 12
experiments["categorical_descriptor_feature"] = (["cd1"] * 6 + ["cd2"] * 6) * 2
experiments["valid_y"] = 1


##### LIST OF TASTE CASES #####

test_cases = []

# CASE 1
test_case = {}
domain = Domain(
    input_features=continuous_input_features,
    output_features=output_features,
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
domain = Domain(
    input_features=continuous_input_features,
    output_features=output_features,
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
    features_used, features_unused = domain.get_nchoosek_combinations()
    for features in test_case["test_features_used"]:
        assert features in features_used
    for features in test_case["test_features_unused"]:
        assert features in features_unused
