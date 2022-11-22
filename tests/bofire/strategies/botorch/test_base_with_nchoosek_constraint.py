from typing import List

import numpy as np
import pandas as pd
import pytest

from bofire.domain.constraints import NChooseKConstraint
from bofire.domain.domain import Domain
from bofire.domain.features import (
    CategoricalDescriptorInput,
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
)
from bofire.strategies.botorch.sobo import AcquisitionFunctionEnum, BoTorchSoboStrategy
from bofire.strategies.strategy import (
    CategoricalEncodingEnum,
    CategoricalMethodEnum,
    DescriptorEncodingEnum,
    DescriptorMethodEnum,
)

# NChooseK constraints 1
cc1a = NChooseKConstraint(features=['0', '1', '2', '3'], min_count=2, max_count=3, none_also_valid=True)
cc2a = NChooseKConstraint(features=['2', '3', '4', '5'], min_count=1, max_count=2, none_also_valid=True)

# NChooseK constraints 2
cc1b = NChooseKConstraint(features=['0', '1', '2', '3'], min_count=2, max_count=3, none_also_valid=False)
cc2b = NChooseKConstraint(features=['2', '3', '4', '5'], min_count=1, max_count=2, none_also_valid=True)

# NChooseK constraint 3
cc3 = NChooseKConstraint(features=['0', '1', '2', '3'], min_count=2, max_count=3, none_also_valid=True)

# input features
continuous_input_features = []
for i in range(6):
    f = ContinuousInput(key=str(i), lower_bound=0, upper_bound=1)
    continuous_input_features.append(f)
categorical_feature = CategoricalInput(key='categorical_feature', categories=['c1', 'c2'])
categorical_descriptor_feature = CategoricalDescriptorInput(key='categorical_descriptor_feature', categories=['cd1', 'cd2'], descriptors=['d1', 'd2'], values=[[1.0, 1.0], [2.0, 2.0]])

# output feature
output_features = [ContinuousOutput(key='y')]


'''
TEST CASES:

CASE 1: 6 continuous features, 2 overlapping NChooseK constraints, none_also_valid: True, True
CASE 2: 6 continuous features, 2 overlapping NChooseK constraints, none_also_valid: False, True

CASE 3: 4 continuous features, 1 NChooseK constraint, none_also_valid: True, 
descriptor_method: EXHAUSTIVE, categorical_method: EXHAUSTIVE, descriptor_encoding: DESCRIPTOR, categorical_encoding: ONE_HOT

CASE 4: 4 continuous features, 1 NChooseK constraint, none_also_valid: True, 
descriptor_method: EXHAUSTIVE, categorical_method: EXHAUSTIVE, descriptor_encoding: CATEGORICAL, categorical_encoding: ONE_HOT

CASE 5: 4 continuous features, 1 NChooseK constraint, none_also_valid: True, 
descriptor_method: EXHAUSTIVE, categorical_method: EXHAUSTIVE, descriptor_encoding: DESCRIPTOR, categorical_encoding: ORDINAL

CASE 6: 4 continuous features, 1 NChooseK constraint, none_also_valid: True, 
descriptor_method: EXHAUSTIVE, categorical_method: EXHAUSTIVE, descriptor_encoding: CATEGORICAL, categorical_encoding: ORDINAL

CASE 7: 4 continuous features, 1 NChooseK constraint, none_also_valid: True, 
descriptor_method: EXHAUSTIVE, categorical_method: FREE, descriptor_encoding: DESCRIPTOR, categorical_encoding: ONE_HOT

CASE 8: 4 continuous features, 1 NChooseK constraint, none_also_valid: True, 
descriptor_method: EXHAUSTIVE, categorical_method: FREE, descriptor_encoding: CATEGORICAL, categorical_encoding: ONE_HOT

CASE 9: 4 continuous features, 1 NChooseK constraint, none_also_valid: True, 
descriptor_method: FREE, categorical_method: EXHAUSTIVE, descriptor_encoding: CATEGORICAL, categorical_encoding: ORDINAL

CASE 10: 4 continuous features, 1 NChooseK constraint, none_also_valid: True, 
descriptor_method: FREE, categorical_method: EXHAUSTIVE, descriptor_encoding: CATEGORICAL, categorical_encoding: ONE_HOT

CASE 11: 4 continuous features, 1 NChooseK constraint, none_also_valid: True, 
descriptor_method: FREE, categorical_method: EXHAUSTIVE, descriptor_encoding: DESCRIPTOR, categorical_encoding: ORDINAL

CASE 12: 4 continuous features, 1 NChooseK constraint, none_also_valid: True, 
descriptor_method: FREE, categorical_method: EXHAUSTIVE, descriptor_encoding: DESCRIPTOR, categorical_encoding: ONE_HOT

CASE 13: 4 continuous features, 1 NChooseK constraint, none_also_valid: True, 
descriptor_method: FREE, categorical_method: FREE, descriptor_encoding: CATEGORICAL, categorical_encoding: ONE_HOT

CASE 14: 4 continuous features, 1 NChooseK constraint, none_also_valid: True, 
descriptor_method: FREE, categorical_method: FREE, descriptor_encoding: DESCRIPTOR, categorical_encoding: ONE_HOT
'''

# CASE 1
test_fixed_values_1 = [
    {3: 0.0, 4: 0.0, 5: 0.0},
    {2: 0.0, 4: 0.0, 5: 0.0},
    {2: 0.0, 3: 0.0, 5: 0.0},
    {2: 0.0, 3: 0.0, 4: 0.0},
    {3: 0.0, 5: 0.0},
    {3: 0.0, 4: 0.0},
    {2: 0.0, 5: 0.0},
    {2: 0.0, 4: 0.0},
    {2: 0.0, 3: 0.0},
    {2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0},
    {1: 0.0, 3: 0.0, 4: 0.0, 5: 0.0},
    {1: 0.0, 4: 0.0, 5: 0.0},
    {1: 0.0, 3: 0.0, 5: 0.0},
    {1: 0.0, 3: 0.0, 4: 0.0},
    {1: 0.0, 2: 0.0, 4: 0.0, 5: 0.0},
    {1: 0.0, 2: 0.0, 5: 0.0},
    {1: 0.0, 2: 0.0, 4: 0.0},
    {0: 0.0, 3: 0.0, 4: 0.0, 5: 0.0},
    {0: 0.0, 4: 0.0, 5: 0.0},
    {0: 0.0, 3: 0.0, 5: 0.0},
    {0: 0.0, 3: 0.0, 4: 0.0},
    {0: 0.0, 2: 0.0, 4: 0.0, 5: 0.0},
    {0: 0.0, 2: 0.0, 5: 0.0},
    {0: 0.0, 2: 0.0, 4: 0.0},
    {0: 0.0, 1: 0.0, 4: 0.0, 5: 0.0},
    {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 5: 0.0},
    {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0},
    {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
    {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}
]

# CASE 2
test_fixed_values_2 = [
    {3: 0.0, 4: 0.0, 5: 0.0},
    {2: 0.0, 4: 0.0, 5: 0.0},
    {2: 0.0, 3: 0.0, 5: 0.0},
    {2: 0.0, 3: 0.0, 4: 0.0},
    {3: 0.0, 5: 0.0},
    {3: 0.0, 4: 0.0},
    {2: 0.0, 5: 0.0},
    {2: 0.0, 4: 0.0},
    {2: 0.0, 3: 0.0},
    {2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0},
    {1: 0.0, 3: 0.0, 4: 0.0, 5: 0.0},
    {1: 0.0, 4: 0.0, 5: 0.0},
    {1: 0.0, 3: 0.0, 5: 0.0},
    {1: 0.0, 3: 0.0, 4: 0.0},
    {1: 0.0, 2: 0.0, 4: 0.0, 5: 0.0},
    {1: 0.0, 2: 0.0, 5: 0.0},
    {1: 0.0, 2: 0.0, 4: 0.0},
    {0: 0.0, 3: 0.0, 4: 0.0, 5: 0.0},
    {0: 0.0, 4: 0.0, 5: 0.0},
    {0: 0.0, 3: 0.0, 5: 0.0},
    {0: 0.0, 3: 0.0, 4: 0.0},
    {0: 0.0, 2: 0.0, 4: 0.0, 5: 0.0},
    {0: 0.0, 2: 0.0, 5: 0.0},
    {0: 0.0, 2: 0.0, 4: 0.0},
    {0: 0.0, 1: 0.0, 4: 0.0, 5: 0.0},
]

# CASE 3
test_fixed_values_3 = [
    {2: 0.0, 3: 0.0, 6: 1.0, 7: 0.0, 8: 1.0, 9: 1.0},
    {1: 0.0, 3: 0.0, 6: 1.0, 7: 0.0, 8: 1.0, 9: 1.0},
    {1: 0.0, 2: 0.0, 6: 1.0, 7: 0.0, 8: 1.0, 9: 1.0},
    {0: 0.0, 3: 0.0, 6: 1.0, 7: 0.0, 8: 1.0, 9: 1.0},
    {0: 0.0, 2: 0.0, 6: 1.0, 7: 0.0, 8: 1.0, 9: 1.0},
    {0: 0.0, 1: 0.0, 6: 1.0, 7: 0.0, 8: 1.0, 9: 1.0},
    {3: 0.0, 6: 1.0, 7: 0.0, 8: 1.0, 9: 1.0},
    {2: 0.0, 6: 1.0, 7: 0.0, 8: 1.0, 9: 1.0},
    {1: 0.0, 6: 1.0, 7: 0.0, 8: 1.0, 9: 1.0},
    {0: 0.0, 6: 1.0, 7: 0.0, 8: 1.0, 9: 1.0},
    {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 6: 1.0, 7: 0.0, 8: 1.0, 9: 1.0},
    {2: 0.0, 3: 0.0, 6: 0.0, 7: 1.0, 8: 1.0, 9: 1.0},
    {1: 0.0, 3: 0.0, 6: 0.0, 7: 1.0, 8: 1.0, 9: 1.0},
    {1: 0.0, 2: 0.0, 6: 0.0, 7: 1.0, 8: 1.0, 9: 1.0},
    {0: 0.0, 3: 0.0, 6: 0.0, 7: 1.0, 8: 1.0, 9: 1.0},
    {0: 0.0, 2: 0.0, 6: 0.0, 7: 1.0, 8: 1.0, 9: 1.0},
    {0: 0.0, 1: 0.0, 6: 0.0, 7: 1.0, 8: 1.0, 9: 1.0},
    {3: 0.0, 6: 0.0, 7: 1.0, 8: 1.0, 9: 1.0},
    {2: 0.0, 6: 0.0, 7: 1.0, 8: 1.0, 9: 1.0},
    {1: 0.0, 6: 0.0, 7: 1.0, 8: 1.0, 9: 1.0},
    {0: 0.0, 6: 0.0, 7: 1.0, 8: 1.0, 9: 1.0},
    {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 6: 0.0, 7: 1.0, 8: 1.0, 9: 1.0},
    {2: 0.0, 3: 0.0, 6: 1.0, 7: 0.0, 8: 2.0, 9: 2.0},
    {1: 0.0, 3: 0.0, 6: 1.0, 7: 0.0, 8: 2.0, 9: 2.0},
    {1: 0.0, 2: 0.0, 6: 1.0, 7: 0.0, 8: 2.0, 9: 2.0},
    {0: 0.0, 3: 0.0, 6: 1.0, 7: 0.0, 8: 2.0, 9: 2.0},
    {0: 0.0, 2: 0.0, 6: 1.0, 7: 0.0, 8: 2.0, 9: 2.0},
    {0: 0.0, 1: 0.0, 6: 1.0, 7: 0.0, 8: 2.0, 9: 2.0},
    {3: 0.0, 6: 1.0, 7: 0.0, 8: 2.0, 9: 2.0},
    {2: 0.0, 6: 1.0, 7: 0.0, 8: 2.0, 9: 2.0},
    {1: 0.0, 6: 1.0, 7: 0.0, 8: 2.0, 9: 2.0},
    {0: 0.0, 6: 1.0, 7: 0.0, 8: 2.0, 9: 2.0},
    {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 6: 1.0, 7: 0.0, 8: 2.0, 9: 2.0},
    {2: 0.0, 3: 0.0, 6: 0.0, 7: 1.0, 8: 2.0, 9: 2.0},
    {1: 0.0, 3: 0.0, 6: 0.0, 7: 1.0, 8: 2.0, 9: 2.0},
    {1: 0.0, 2: 0.0, 6: 0.0, 7: 1.0, 8: 2.0, 9: 2.0},
    {0: 0.0, 3: 0.0, 6: 0.0, 7: 1.0, 8: 2.0, 9: 2.0},
    {0: 0.0, 2: 0.0, 6: 0.0, 7: 1.0, 8: 2.0, 9: 2.0},
    {0: 0.0, 1: 0.0, 6: 0.0, 7: 1.0, 8: 2.0, 9: 2.0},
    {3: 0.0, 6: 0.0, 7: 1.0, 8: 2.0, 9: 2.0},
    {2: 0.0, 6: 0.0, 7: 1.0, 8: 2.0, 9: 2.0},
    {1: 0.0, 6: 0.0, 7: 1.0, 8: 2.0, 9: 2.0},
    {0: 0.0, 6: 0.0, 7: 1.0, 8: 2.0, 9: 2.0},
    {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 6: 0.0, 7: 1.0, 8: 2.0, 9: 2.0}
]

test_fixed_values_4 = [
    {2: 0.0, 3: 0.0, 6: 1.0, 7: 0.0, 8: 1.0, 9: 0.0},
    {1: 0.0, 3: 0.0, 6: 1.0, 7: 0.0, 8: 1.0, 9: 0.0},
    {1: 0.0, 2: 0.0, 6: 1.0, 7: 0.0, 8: 1.0, 9: 0.0},
    {0: 0.0, 3: 0.0, 6: 1.0, 7: 0.0, 8: 1.0, 9: 0.0},
    {0: 0.0, 2: 0.0, 6: 1.0, 7: 0.0, 8: 1.0, 9: 0.0},
    {0: 0.0, 1: 0.0, 6: 1.0, 7: 0.0, 8: 1.0, 9: 0.0},
    {3: 0.0, 6: 1.0, 7: 0.0, 8: 1.0, 9: 0.0},
    {2: 0.0, 6: 1.0, 7: 0.0, 8: 1.0, 9: 0.0},
    {1: 0.0, 6: 1.0, 7: 0.0, 8: 1.0, 9: 0.0},
    {0: 0.0, 6: 1.0, 7: 0.0, 8: 1.0, 9: 0.0},
    {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 6: 1.0, 7: 0.0, 8: 1.0, 9: 0.0},
    {2: 0.0, 3: 0.0, 6: 0.0, 7: 1.0, 8: 1.0, 9: 0.0},
    {1: 0.0, 3: 0.0, 6: 0.0, 7: 1.0, 8: 1.0, 9: 0.0},
    {1: 0.0, 2: 0.0, 6: 0.0, 7: 1.0, 8: 1.0, 9: 0.0},
    {0: 0.0, 3: 0.0, 6: 0.0, 7: 1.0, 8: 1.0, 9: 0.0},
    {0: 0.0, 2: 0.0, 6: 0.0, 7: 1.0, 8: 1.0, 9: 0.0},
    {0: 0.0, 1: 0.0, 6: 0.0, 7: 1.0, 8: 1.0, 9: 0.0},
    {3: 0.0, 6: 0.0, 7: 1.0, 8: 1.0, 9: 0.0},
    {2: 0.0, 6: 0.0, 7: 1.0, 8: 1.0, 9: 0.0},
    {1: 0.0, 6: 0.0, 7: 1.0, 8: 1.0, 9: 0.0},
    {0: 0.0, 6: 0.0, 7: 1.0, 8: 1.0, 9: 0.0},
    {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 6: 0.0, 7: 1.0, 8: 1.0, 9: 0.0},
    {2: 0.0, 3: 0.0, 6: 1.0, 7: 0.0, 8: 0.0, 9: 1.0},
    {1: 0.0, 3: 0.0, 6: 1.0, 7: 0.0, 8: 0.0, 9: 1.0},
    {1: 0.0, 2: 0.0, 6: 1.0, 7: 0.0, 8: 0.0, 9: 1.0},
    {0: 0.0, 3: 0.0, 6: 1.0, 7: 0.0, 8: 0.0, 9: 1.0},
    {0: 0.0, 2: 0.0, 6: 1.0, 7: 0.0, 8: 0.0, 9: 1.0},
    {0: 0.0, 1: 0.0, 6: 1.0, 7: 0.0, 8: 0.0, 9: 1.0},
    {3: 0.0, 6: 1.0, 7: 0.0, 8: 0.0, 9: 1.0},
    {2: 0.0, 6: 1.0, 7: 0.0, 8: 0.0, 9: 1.0},
    {1: 0.0, 6: 1.0, 7: 0.0, 8: 0.0, 9: 1.0},
    {0: 0.0, 6: 1.0, 7: 0.0, 8: 0.0, 9: 1.0},
    {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 6: 1.0, 7: 0.0, 8: 0.0, 9: 1.0},
    {2: 0.0, 3: 0.0, 6: 0.0, 7: 1.0, 8: 0.0, 9: 1.0},
    {1: 0.0, 3: 0.0, 6: 0.0, 7: 1.0, 8: 0.0, 9: 1.0},
    {1: 0.0, 2: 0.0, 6: 0.0, 7: 1.0, 8: 0.0, 9: 1.0},
    {0: 0.0, 3: 0.0, 6: 0.0, 7: 1.0, 8: 0.0, 9: 1.0},
    {0: 0.0, 2: 0.0, 6: 0.0, 7: 1.0, 8: 0.0, 9: 1.0},
    {0: 0.0, 1: 0.0, 6: 0.0, 7: 1.0, 8: 0.0, 9: 1.0},
    {3: 0.0, 6: 0.0, 7: 1.0, 8: 0.0, 9: 1.0},
    {2: 0.0, 6: 0.0, 7: 1.0, 8: 0.0, 9: 1.0},
    {1: 0.0, 6: 0.0, 7: 1.0, 8: 0.0, 9: 1.0},
    {0: 0.0, 6: 0.0, 7: 1.0, 8: 0.0, 9: 1.0},
    {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 6: 0.0, 7: 1.0, 8: 0.0, 9: 1.0},
]

test_fixed_values_5 = [
    {2: 0.0, 3: 0.0, 6: 0.0, 7: 1.0, 8: 1.0},
    {1: 0.0, 3: 0.0, 6: 0.0, 7: 1.0, 8: 1.0},
    {1: 0.0, 2: 0.0, 6: 0.0, 7: 1.0, 8: 1.0},
    {0: 0.0, 3: 0.0, 6: 0.0, 7: 1.0, 8: 1.0},
    {0: 0.0, 2: 0.0, 6: 0.0, 7: 1.0, 8: 1.0},
    {0: 0.0, 1: 0.0, 6: 0.0, 7: 1.0, 8: 1.0},
    {3: 0.0, 6: 0.0, 7: 1.0, 8: 1.0},
    {2: 0.0, 6: 0.0, 7: 1.0, 8: 1.0},
    {1: 0.0, 6: 0.0, 7: 1.0, 8: 1.0},
    {0: 0.0, 6: 0.0, 7: 1.0, 8: 1.0},
    {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 6: 0.0, 7: 1.0, 8: 1.0},
    {2: 0.0, 3: 0.0, 6: 1.0, 7: 1.0, 8: 1.0},
    {1: 0.0, 3: 0.0, 6: 1.0, 7: 1.0, 8: 1.0},
    {1: 0.0, 2: 0.0, 6: 1.0, 7: 1.0, 8: 1.0},
    {0: 0.0, 3: 0.0, 6: 1.0, 7: 1.0, 8: 1.0},
    {0: 0.0, 2: 0.0, 6: 1.0, 7: 1.0, 8: 1.0},
    {0: 0.0, 1: 0.0, 6: 1.0, 7: 1.0, 8: 1.0},
    {3: 0.0, 6: 1.0, 7: 1.0, 8: 1.0},
    {2: 0.0, 6: 1.0, 7: 1.0, 8: 1.0},
    {1: 0.0, 6: 1.0, 7: 1.0, 8: 1.0},
    {0: 0.0, 6: 1.0, 7: 1.0, 8: 1.0},
    {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 6: 1.0, 7: 1.0, 8: 1.0},
    {2: 0.0, 3: 0.0, 6: 0.0, 7: 2.0, 8: 2.0},
    {1: 0.0, 3: 0.0, 6: 0.0, 7: 2.0, 8: 2.0},
    {1: 0.0, 2: 0.0, 6: 0.0, 7: 2.0, 8: 2.0},
    {0: 0.0, 3: 0.0, 6: 0.0, 7: 2.0, 8: 2.0},
    {0: 0.0, 2: 0.0, 6: 0.0, 7: 2.0, 8: 2.0},
    {0: 0.0, 1: 0.0, 6: 0.0, 7: 2.0, 8: 2.0},
    {3: 0.0, 6: 0.0, 7: 2.0, 8: 2.0},
    {2: 0.0, 6: 0.0, 7: 2.0, 8: 2.0},
    {1: 0.0, 6: 0.0, 7: 2.0, 8: 2.0},
    {0: 0.0, 6: 0.0, 7: 2.0, 8: 2.0},
    {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 6: 0.0, 7: 2.0, 8: 2.0},
    {2: 0.0, 3: 0.0, 6: 1.0, 7: 2.0, 8: 2.0},
    {1: 0.0, 3: 0.0, 6: 1.0, 7: 2.0, 8: 2.0},
    {1: 0.0, 2: 0.0, 6: 1.0, 7: 2.0, 8: 2.0},
    {0: 0.0, 3: 0.0, 6: 1.0, 7: 2.0, 8: 2.0},
    {0: 0.0, 2: 0.0, 6: 1.0, 7: 2.0, 8: 2.0},
    {0: 0.0, 1: 0.0, 6: 1.0, 7: 2.0, 8: 2.0},
    {3: 0.0, 6: 1.0, 7: 2.0, 8: 2.0},
    {2: 0.0, 6: 1.0, 7: 2.0, 8: 2.0},
    {1: 0.0, 6: 1.0, 7: 2.0, 8: 2.0},
    {0: 0.0, 6: 1.0, 7: 2.0, 8: 2.0},
    {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 6: 1.0, 7: 2.0, 8: 2.0}
]

test_fixed_values_6 = [
    {2: 0.0, 3: 0.0, 6: 0.0, 7: 0.0},
    {1: 0.0, 3: 0.0, 6: 0.0, 7: 0.0},
    {1: 0.0, 2: 0.0, 6: 0.0, 7: 0.0},
    {0: 0.0, 3: 0.0, 6: 0.0, 7: 0.0},
    {0: 0.0, 2: 0.0, 6: 0.0, 7: 0.0},
    {0: 0.0, 1: 0.0, 6: 0.0, 7: 0.0},
    {3: 0.0, 6: 0.0, 7: 0.0},
    {2: 0.0, 6: 0.0, 7: 0.0},
    {1: 0.0, 6: 0.0, 7: 0.0},
    {0: 0.0, 6: 0.0, 7: 0.0},
    {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 6: 0.0, 7: 0.0},
    {2: 0.0, 3: 0.0, 6: 1.0, 7: 0.0},
    {1: 0.0, 3: 0.0, 6: 1.0, 7: 0.0},
    {1: 0.0, 2: 0.0, 6: 1.0, 7: 0.0},
    {0: 0.0, 3: 0.0, 6: 1.0, 7: 0.0},
    {0: 0.0, 2: 0.0, 6: 1.0, 7: 0.0},
    {0: 0.0, 1: 0.0, 6: 1.0, 7: 0.0},
    {3: 0.0, 6: 1.0, 7: 0.0},
    {2: 0.0, 6: 1.0, 7: 0.0},
    {1: 0.0, 6: 1.0, 7: 0.0},
    {0: 0.0, 6: 1.0, 7: 0.0},
    {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 6: 1.0, 7: 0.0},
    {2: 0.0, 3: 0.0, 6: 0.0, 7: 1.0},
    {1: 0.0, 3: 0.0, 6: 0.0, 7: 1.0},
    {1: 0.0, 2: 0.0, 6: 0.0, 7: 1.0},
    {0: 0.0, 3: 0.0, 6: 0.0, 7: 1.0},
    {0: 0.0, 2: 0.0, 6: 0.0, 7: 1.0},
    {0: 0.0, 1: 0.0, 6: 0.0, 7: 1.0},
    {3: 0.0, 6: 0.0, 7: 1.0},
    {2: 0.0, 6: 0.0, 7: 1.0},
    {1: 0.0, 6: 0.0, 7: 1.0},
    {0: 0.0, 6: 0.0, 7: 1.0},
    {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 6: 0.0, 7: 1.0},
    {2: 0.0, 3: 0.0, 6: 1.0, 7: 1.0},
    {1: 0.0, 3: 0.0, 6: 1.0, 7: 1.0},
    {1: 0.0, 2: 0.0, 6: 1.0, 7: 1.0},
    {0: 0.0, 3: 0.0, 6: 1.0, 7: 1.0},
    {0: 0.0, 2: 0.0, 6: 1.0, 7: 1.0},
    {0: 0.0, 1: 0.0, 6: 1.0, 7: 1.0},
    {3: 0.0, 6: 1.0, 7: 1.0},
    {2: 0.0, 6: 1.0, 7: 1.0},
    {1: 0.0, 6: 1.0, 7: 1.0},
    {0: 0.0, 6: 1.0, 7: 1.0},
    {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 6: 1.0, 7: 1.0}
]

test_fixed_values_7 = [
    {2: 0.0, 3: 0.0, 8: 1.0, 9: 1.0},
    {1: 0.0, 3: 0.0, 8: 1.0, 9: 1.0},
    {1: 0.0, 2: 0.0, 8: 1.0, 9: 1.0},
    {0: 0.0, 3: 0.0, 8: 1.0, 9: 1.0},
    {0: 0.0, 2: 0.0, 8: 1.0, 9: 1.0},
    {0: 0.0, 1: 0.0, 8: 1.0, 9: 1.0},
    {3: 0.0, 8: 1.0, 9: 1.0},
    {2: 0.0, 8: 1.0, 9: 1.0},
    {1: 0.0, 8: 1.0, 9: 1.0},
    {0: 0.0, 8: 1.0, 9: 1.0},
    {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 8: 1.0, 9: 1.0},
    {2: 0.0, 3: 0.0, 8: 2.0, 9: 2.0},
    {1: 0.0, 3: 0.0, 8: 2.0, 9: 2.0},
    {1: 0.0, 2: 0.0, 8: 2.0, 9: 2.0},
    {0: 0.0, 3: 0.0, 8: 2.0, 9: 2.0},
    {0: 0.0, 2: 0.0, 8: 2.0, 9: 2.0},
    {0: 0.0, 1: 0.0, 8: 2.0, 9: 2.0},
    {3: 0.0, 8: 2.0, 9: 2.0},
    {2: 0.0, 8: 2.0, 9: 2.0},
    {1: 0.0, 8: 2.0, 9: 2.0},
    {0: 0.0, 8: 2.0, 9: 2.0},
    {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 8: 2.0, 9: 2.0}
    
]

test_fixed_values_8 = [
    {2: 0.0, 3: 0.0},
    {1: 0.0, 3: 0.0},
    {1: 0.0, 2: 0.0},
    {0: 0.0, 3: 0.0},
    {0: 0.0, 2: 0.0},
    {0: 0.0, 1: 0.0},
    {3: 0.0},
    {2: 0.0},
    {1: 0.0},
    {0: 0.0},
    {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
]

test_fixed_values_9 = [
    {2: 0.0, 3: 0.0, 6: 0.0, 7: 0.0},
    {1: 0.0, 3: 0.0, 6: 0.0, 7: 0.0},
    {1: 0.0, 2: 0.0, 6: 0.0, 7: 0.0},
    {0: 0.0, 3: 0.0, 6: 0.0, 7: 0.0},
    {0: 0.0, 2: 0.0, 6: 0.0, 7: 0.0},
    {0: 0.0, 1: 0.0, 6: 0.0, 7: 0.0},
    {3: 0.0, 6: 0.0, 7: 0.0},
    {2: 0.0, 6: 0.0, 7: 0.0},
    {1: 0.0, 6: 0.0, 7: 0.0},
    {0: 0.0, 6: 0.0, 7: 0.0},
    {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 6: 0.0, 7: 0.0},
    {2: 0.0, 3: 0.0, 6: 1.0, 7: 0.0},
    {1: 0.0, 3: 0.0, 6: 1.0, 7: 0.0},
    {1: 0.0, 2: 0.0, 6: 1.0, 7: 0.0},
    {0: 0.0, 3: 0.0, 6: 1.0, 7: 0.0},
    {0: 0.0, 2: 0.0, 6: 1.0, 7: 0.0},
    {0: 0.0, 1: 0.0, 6: 1.0, 7: 0.0},
    {3: 0.0, 6: 1.0, 7: 0.0},
    {2: 0.0, 6: 1.0, 7: 0.0},
    {1: 0.0, 6: 1.0, 7: 0.0},
    {0: 0.0, 6: 1.0, 7: 0.0},
    {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 6: 1.0, 7: 0.0},
    {2: 0.0, 3: 0.0, 6: 0.0, 7: 1.0},
    {1: 0.0, 3: 0.0, 6: 0.0, 7: 1.0},
    {1: 0.0, 2: 0.0, 6: 0.0, 7: 1.0},
    {0: 0.0, 3: 0.0, 6: 0.0, 7: 1.0},
    {0: 0.0, 2: 0.0, 6: 0.0, 7: 1.0},
    {0: 0.0, 1: 0.0, 6: 0.0, 7: 1.0},
    {3: 0.0, 6: 0.0, 7: 1.0},
    {2: 0.0, 6: 0.0, 7: 1.0},
    {1: 0.0, 6: 0.0, 7: 1.0},
    {0: 0.0, 6: 0.0, 7: 1.0},
    {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 6: 0.0, 7: 1.0},
    {2: 0.0, 3: 0.0, 6: 1.0, 7: 1.0},
    {1: 0.0, 3: 0.0, 6: 1.0, 7: 1.0},
    {1: 0.0, 2: 0.0, 6: 1.0, 7: 1.0},
    {0: 0.0, 3: 0.0, 6: 1.0, 7: 1.0},
    {0: 0.0, 2: 0.0, 6: 1.0, 7: 1.0},
    {0: 0.0, 1: 0.0, 6: 1.0, 7: 1.0},
    {3: 0.0, 6: 1.0, 7: 1.0},
    {2: 0.0, 6: 1.0, 7: 1.0},
    {1: 0.0, 6: 1.0, 7: 1.0},
    {0: 0.0, 6: 1.0, 7: 1.0},
    {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 6: 1.0, 7: 1.0}
]

test_fixed_values_10 = [
    {2: 0.0, 3: 0.0, 6: 1.0, 7: 0.0, 8: 1.0, 9: 0.0},
    {1: 0.0, 3: 0.0, 6: 1.0, 7: 0.0, 8: 1.0, 9: 0.0},
    {1: 0.0, 2: 0.0, 6: 1.0, 7: 0.0, 8: 1.0, 9: 0.0},
    {0: 0.0, 3: 0.0, 6: 1.0, 7: 0.0, 8: 1.0, 9: 0.0},
    {0: 0.0, 2: 0.0, 6: 1.0, 7: 0.0, 8: 1.0, 9: 0.0},
    {0: 0.0, 1: 0.0, 6: 1.0, 7: 0.0, 8: 1.0, 9: 0.0},
    {3: 0.0, 6: 1.0, 7: 0.0, 8: 1.0, 9: 0.0},
    {2: 0.0, 6: 1.0, 7: 0.0, 8: 1.0, 9: 0.0},
    {1: 0.0, 6: 1.0, 7: 0.0, 8: 1.0, 9: 0.0},
    {0: 0.0, 6: 1.0, 7: 0.0, 8: 1.0, 9: 0.0},
    {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 6: 1.0, 7: 0.0, 8: 1.0, 9: 0.0},
    {2: 0.0, 3: 0.0, 6: 0.0, 7: 1.0, 8: 1.0, 9: 0.0},
    {1: 0.0, 3: 0.0, 6: 0.0, 7: 1.0, 8: 1.0, 9: 0.0},
    {1: 0.0, 2: 0.0, 6: 0.0, 7: 1.0, 8: 1.0, 9: 0.0},
    {0: 0.0, 3: 0.0, 6: 0.0, 7: 1.0, 8: 1.0, 9: 0.0},
    {0: 0.0, 2: 0.0, 6: 0.0, 7: 1.0, 8: 1.0, 9: 0.0},
    {0: 0.0, 1: 0.0, 6: 0.0, 7: 1.0, 8: 1.0, 9: 0.0},
    {3: 0.0, 6: 0.0, 7: 1.0, 8: 1.0, 9: 0.0},
    {2: 0.0, 6: 0.0, 7: 1.0, 8: 1.0, 9: 0.0},
    {1: 0.0, 6: 0.0, 7: 1.0, 8: 1.0, 9: 0.0},
    {0: 0.0, 6: 0.0, 7: 1.0, 8: 1.0, 9: 0.0},
    {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 6: 0.0, 7: 1.0, 8: 1.0, 9: 0.0},
    {2: 0.0, 3: 0.0, 6: 1.0, 7: 0.0, 8: 0.0, 9: 1.0},
    {1: 0.0, 3: 0.0, 6: 1.0, 7: 0.0, 8: 0.0, 9: 1.0},
    {1: 0.0, 2: 0.0, 6: 1.0, 7: 0.0, 8: 0.0, 9: 1.0},
    {0: 0.0, 3: 0.0, 6: 1.0, 7: 0.0, 8: 0.0, 9: 1.0},
    {0: 0.0, 2: 0.0, 6: 1.0, 7: 0.0, 8: 0.0, 9: 1.0},
    {0: 0.0, 1: 0.0, 6: 1.0, 7: 0.0, 8: 0.0, 9: 1.0},
    {3: 0.0, 6: 1.0, 7: 0.0, 8: 0.0, 9: 1.0},
    {2: 0.0, 6: 1.0, 7: 0.0, 8: 0.0, 9: 1.0},
    {1: 0.0, 6: 1.0, 7: 0.0, 8: 0.0, 9: 1.0},
    {0: 0.0, 6: 1.0, 7: 0.0, 8: 0.0, 9: 1.0},
    {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 6: 1.0, 7: 0.0, 8: 0.0, 9: 1.0},
    {2: 0.0, 3: 0.0, 6: 0.0, 7: 1.0, 8: 0.0, 9: 1.0},
    {1: 0.0, 3: 0.0, 6: 0.0, 7: 1.0, 8: 0.0, 9: 1.0},
    {1: 0.0, 2: 0.0, 6: 0.0, 7: 1.0, 8: 0.0, 9: 1.0},
    {0: 0.0, 3: 0.0, 6: 0.0, 7: 1.0, 8: 0.0, 9: 1.0},
    {0: 0.0, 2: 0.0, 6: 0.0, 7: 1.0, 8: 0.0, 9: 1.0},
    {0: 0.0, 1: 0.0, 6: 0.0, 7: 1.0, 8: 0.0, 9: 1.0},
    {3: 0.0, 6: 0.0, 7: 1.0, 8: 0.0, 9: 1.0},
    {2: 0.0, 6: 0.0, 7: 1.0, 8: 0.0, 9: 1.0},
    {1: 0.0, 6: 0.0, 7: 1.0, 8: 0.0, 9: 1.0},
    {0: 0.0, 6: 0.0, 7: 1.0, 8: 0.0, 9: 1.0},
    {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 6: 0.0, 7: 1.0, 8: 0.0, 9: 1.0},
]

test_fixed_values_11 = [
    {2: 0.0, 3: 0.0, 6: 0.0},
    {1: 0.0, 3: 0.0, 6: 0.0},
    {1: 0.0, 2: 0.0, 6: 0.0},
    {0: 0.0, 3: 0.0, 6: 0.0},
    {0: 0.0, 2: 0.0, 6: 0.0},
    {0: 0.0, 1: 0.0, 6: 0.0},
    {3: 0.0, 6: 0.0},
    {2: 0.0, 6: 0.0},
    {1: 0.0, 6: 0.0},
    {0: 0.0, 6: 0.0},
    {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 6: 0.0},
    {2: 0.0, 3: 0.0, 6: 1.0},
    {1: 0.0, 3: 0.0, 6: 1.0},
    {1: 0.0, 2: 0.0, 6: 1.0},
    {0: 0.0, 3: 0.0, 6: 1.0},
    {0: 0.0, 2: 0.0, 6: 1.0},
    {0: 0.0, 1: 0.0, 6: 1.0},
    {3: 0.0, 6: 1.0},
    {2: 0.0, 6: 1.0},
    {1: 0.0, 6: 1.0},
    {0: 0.0, 6: 1.0},
    {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 6: 1.0},
]

test_fixed_values_12 = [
    {2: 0.0, 3: 0.0, 6: 1.0, 7: 0.0},
    {1: 0.0, 3: 0.0, 6: 1.0, 7: 0.0},
    {1: 0.0, 2: 0.0, 6: 1.0, 7: 0.0},
    {0: 0.0, 3: 0.0, 6: 1.0, 7: 0.0},
    {0: 0.0, 2: 0.0, 6: 1.0, 7: 0.0},
    {0: 0.0, 1: 0.0, 6: 1.0, 7: 0.0},
    {3: 0.0, 6: 1.0, 7: 0.0},
    {2: 0.0, 6: 1.0, 7: 0.0},
    {1: 0.0, 6: 1.0, 7: 0.0},
    {0: 0.0, 6: 1.0, 7: 0.0},
    {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 6: 1.0, 7: 0.0},
    {2: 0.0, 3: 0.0, 6: 0.0, 7: 1.0},
    {1: 0.0, 3: 0.0, 6: 0.0, 7: 1.0},
    {1: 0.0, 2: 0.0, 6: 0.0, 7: 1.0},
    {0: 0.0, 3: 0.0, 6: 0.0, 7: 1.0},
    {0: 0.0, 2: 0.0, 6: 0.0, 7: 1.0},
    {0: 0.0, 1: 0.0, 6: 0.0, 7: 1.0},
    {3: 0.0, 6: 0.0, 7: 1.0},
    {2: 0.0, 6: 0.0, 7: 1.0},
    {1: 0.0, 6: 0.0, 7: 1.0},
    {0: 0.0, 6: 0.0, 7: 1.0},
    {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 6: 0.0, 7: 1.0}
]

test_fixed_values_13 = [
    {2: 0.0, 3: 0.0},
    {1: 0.0, 3: 0.0},
    {1: 0.0, 2: 0.0},
    {0: 0.0, 3: 0.0},
    {0: 0.0, 2: 0.0},
    {0: 0.0, 1: 0.0},
    {3: 0.0},
    {2: 0.0},
    {1: 0.0},
    {0: 0.0},
    {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
]

test_fixed_values_14 = [
    {2: 0.0, 3: 0.0},
    {1: 0.0, 3: 0.0},
    {1: 0.0, 2: 0.0},
    {0: 0.0, 3: 0.0},
    {0: 0.0, 2: 0.0},
    {0: 0.0, 1: 0.0},
    {3: 0.0},
    {2: 0.0},
    {1: 0.0},
    {0: 0.0},
    {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
]

# experiments
experiments = pd.DataFrame(np.random.uniform(size=(24, 7)), columns=['0', '1', '2', '3', '4', '5', 'y'])
experiments['categorical_feature'] = ['c1']*12 + ['c2']*12
experiments['categorical_descriptor_feature'] = (['cd1']*6 + ['cd2']*6)*2
experiments['valid_y'] = 1





##### LIST OF TASTE CASES #####

test_cases = []

# CASE 1
test_case = {}
domain = Domain(
    input_features=continuous_input_features, 
    output_features=output_features, 
    constraints=[cc1a, cc2a]
    )
test_case['domain'] = domain
test_case['experiments'] = experiments
test_case['descriptor_method'] = DescriptorMethodEnum.EXHAUSTIVE
test_case['categorical_method'] = CategoricalMethodEnum.EXHAUSTIVE
test_case['descriptor_encoding'] = DescriptorEncodingEnum.DESCRIPTOR
test_case['categorical_encoding'] = CategoricalEncodingEnum.ONE_HOT
test_case['test_fixed_values'] = test_fixed_values_1
test_cases.append(test_case)

# CASE 2
test_case = {}
domain = Domain(
    input_features=continuous_input_features, 
    output_features=output_features, 
    constraints=[cc1b, cc2b]
    )
test_case['domain'] = domain
test_case['experiments'] = experiments
test_case['descriptor_method'] = DescriptorMethodEnum.EXHAUSTIVE
test_case['categorical_method'] = CategoricalMethodEnum.EXHAUSTIVE
test_case['descriptor_encoding'] = DescriptorEncodingEnum.DESCRIPTOR
test_case['categorical_encoding'] = CategoricalEncodingEnum.ONE_HOT
test_case['test_fixed_values'] = test_fixed_values_2
test_cases.append(test_case)

# CASE 3
test_case = {}
domain = Domain(
    input_features=continuous_input_features + [categorical_feature, categorical_descriptor_feature], 
    output_features=output_features, 
    constraints=[cc3]
    )
test_case['domain'] = domain
test_case['experiments'] = experiments
test_case['descriptor_method'] = DescriptorMethodEnum.EXHAUSTIVE
test_case['categorical_method'] = CategoricalMethodEnum.EXHAUSTIVE
test_case['descriptor_encoding'] = DescriptorEncodingEnum.DESCRIPTOR
test_case['categorical_encoding'] = CategoricalEncodingEnum.ONE_HOT
test_case['test_fixed_values'] = test_fixed_values_3
test_cases.append(test_case)

# CASE 4
test_case = {}
domain = Domain(
    input_features=continuous_input_features + [categorical_feature, categorical_descriptor_feature], 
    output_features=output_features, 
    constraints=[cc3]
    )
test_case['domain'] = domain
test_case['experiments'] = experiments
test_case['descriptor_method'] = DescriptorMethodEnum.EXHAUSTIVE
test_case['categorical_method'] = CategoricalMethodEnum.EXHAUSTIVE
test_case['descriptor_encoding'] = DescriptorEncodingEnum.CATEGORICAL
test_case['categorical_encoding'] = CategoricalEncodingEnum.ONE_HOT
test_case['test_fixed_values'] = test_fixed_values_4
test_cases.append(test_case)

# CASE 5
test_case = {}
domain = Domain(
    input_features=continuous_input_features + [categorical_feature, categorical_descriptor_feature], 
    output_features=output_features, 
    constraints=[cc3]
    )
test_case['domain'] = domain
test_case['experiments'] = experiments
test_case['descriptor_method'] = DescriptorMethodEnum.EXHAUSTIVE
test_case['categorical_method'] = CategoricalMethodEnum.EXHAUSTIVE
test_case['descriptor_encoding'] = DescriptorEncodingEnum.DESCRIPTOR
test_case['categorical_encoding'] = CategoricalEncodingEnum.ORDINAL
test_case['test_fixed_values'] = test_fixed_values_5
test_cases.append(test_case)

# CASE 6
test_case = {}
domain = Domain(
    input_features=continuous_input_features + [categorical_feature, categorical_descriptor_feature], 
    output_features=output_features, 
    constraints=[cc3]
    )
test_case['domain'] = domain
test_case['experiments'] = experiments
test_case['descriptor_method'] = DescriptorMethodEnum.EXHAUSTIVE
test_case['categorical_method'] = CategoricalMethodEnum.EXHAUSTIVE
test_case['descriptor_encoding'] = DescriptorEncodingEnum.CATEGORICAL
test_case['categorical_encoding'] = CategoricalEncodingEnum.ORDINAL
test_case['test_fixed_values'] = test_fixed_values_6
test_cases.append(test_case)

# CASE 7
test_case = {}
domain = Domain(
    input_features=continuous_input_features + [categorical_feature, categorical_descriptor_feature], 
    output_features=output_features, 
    constraints=[cc3]
    )
test_case['domain'] = domain
test_case['experiments'] = experiments
test_case['descriptor_method'] = DescriptorMethodEnum.EXHAUSTIVE
test_case['categorical_method'] = CategoricalMethodEnum.FREE
test_case['descriptor_encoding'] = DescriptorEncodingEnum.DESCRIPTOR
test_case['categorical_encoding'] = CategoricalEncodingEnum.ONE_HOT
test_case['test_fixed_values'] = test_fixed_values_7
test_cases.append(test_case)

# CASE 8
test_case = {}
domain = Domain(
    input_features=continuous_input_features + [categorical_feature, categorical_descriptor_feature], 
    output_features=output_features, 
    constraints=[cc3]
    )
test_case['domain'] = domain
test_case['experiments'] = experiments
test_case['descriptor_method'] = DescriptorMethodEnum.EXHAUSTIVE
test_case['categorical_method'] = CategoricalMethodEnum.FREE
test_case['descriptor_encoding'] = DescriptorEncodingEnum.CATEGORICAL
test_case['categorical_encoding'] = CategoricalEncodingEnum.ONE_HOT
test_case['test_fixed_values'] = test_fixed_values_8
test_cases.append(test_case)

# CASE 9
test_case = {}
domain = Domain(
    input_features=continuous_input_features + [categorical_feature, categorical_descriptor_feature], 
    output_features=output_features, 
    constraints=[cc3]
    )
test_case['domain'] = domain
test_case['experiments'] = experiments
test_case['descriptor_method'] = DescriptorMethodEnum.FREE
test_case['categorical_method'] = CategoricalMethodEnum.EXHAUSTIVE
test_case['descriptor_encoding'] = DescriptorEncodingEnum.CATEGORICAL
test_case['categorical_encoding'] = CategoricalEncodingEnum.ORDINAL
test_case['test_fixed_values'] = test_fixed_values_9
test_cases.append(test_case)

# CASE 10
test_case = {}
domain = Domain(
    input_features=continuous_input_features + [categorical_feature, categorical_descriptor_feature], 
    output_features=output_features, 
    constraints=[cc3]
    )
test_case['domain'] = domain
test_case['experiments'] = experiments
test_case['descriptor_method'] = DescriptorMethodEnum.FREE
test_case['categorical_method'] = CategoricalMethodEnum.EXHAUSTIVE
test_case['descriptor_encoding'] = DescriptorEncodingEnum.CATEGORICAL
test_case['categorical_encoding'] = CategoricalEncodingEnum.ONE_HOT
test_case['test_fixed_values'] = test_fixed_values_10
test_cases.append(test_case)

# CASE 11
test_case = {}
domain = Domain(
    input_features=continuous_input_features + [categorical_feature, categorical_descriptor_feature], 
    output_features=output_features, 
    constraints=[cc3]
    )
test_case['domain'] = domain
test_case['experiments'] = experiments
test_case['descriptor_method'] = DescriptorMethodEnum.FREE
test_case['categorical_method'] = CategoricalMethodEnum.EXHAUSTIVE
test_case['descriptor_encoding'] = DescriptorEncodingEnum.DESCRIPTOR
test_case['categorical_encoding'] = CategoricalEncodingEnum.ORDINAL
test_case['test_fixed_values'] = test_fixed_values_11
test_cases.append(test_case)

# CASE 12
test_case = {}
domain = Domain(
    input_features=continuous_input_features + [categorical_feature, categorical_descriptor_feature], 
    output_features=output_features, 
    constraints=[cc3]
    )
test_case['domain'] = domain
test_case['experiments'] = experiments
test_case['descriptor_method'] = DescriptorMethodEnum.FREE
test_case['categorical_method'] = CategoricalMethodEnum.EXHAUSTIVE
test_case['descriptor_encoding'] = DescriptorEncodingEnum.DESCRIPTOR
test_case['categorical_encoding'] = CategoricalEncodingEnum.ONE_HOT
test_case['test_fixed_values'] = test_fixed_values_12
test_cases.append(test_case)

# CASE 13
test_case = {}
domain = Domain(
    input_features=continuous_input_features + [categorical_feature, categorical_descriptor_feature], 
    output_features=output_features, 
    constraints=[cc3]
    )
test_case['domain'] = domain
test_case['experiments'] = experiments
test_case['descriptor_method'] = DescriptorMethodEnum.FREE
test_case['categorical_method'] = CategoricalMethodEnum.FREE
test_case['descriptor_encoding'] = DescriptorEncodingEnum.CATEGORICAL
test_case['categorical_encoding'] = CategoricalEncodingEnum.ONE_HOT
test_case['test_fixed_values'] = test_fixed_values_13
test_cases.append(test_case)

# CASE 14
test_case = {}
domain = Domain(
    input_features=continuous_input_features + [categorical_feature, categorical_descriptor_feature], 
    output_features=output_features, 
    constraints=[cc3]
    )
test_case['domain'] = domain
test_case['experiments'] = experiments
test_case['descriptor_method'] = DescriptorMethodEnum.FREE
test_case['categorical_method'] = CategoricalMethodEnum.FREE
test_case['descriptor_encoding'] = DescriptorEncodingEnum.DESCRIPTOR
test_case['categorical_encoding'] = CategoricalEncodingEnum.ONE_HOT
test_case['test_fixed_values'] = test_fixed_values_14
test_cases.append(test_case)




@pytest.mark.parametrize("test_case", test_cases)
def test_concurrency_fixed_values(test_case):
    sobo = BoTorchSoboStrategy(
        domain=test_case['domain'],
        experiments=experiments, 
        acquisition_function=AcquisitionFunctionEnum.QNEI,
        descriptor_method=test_case['descriptor_method'],
        categorical_method=test_case['categorical_method'],
        descriptor_encoding=test_case['descriptor_encoding'],
        categorical_encoding=test_case['categorical_encoding'],
        seed=0)
    fixed_values = sobo.get_fixed_values_list()
    # assert len(fixed_values) == len(test_case['test_fixed_values'])
    for features in test_case['test_fixed_values']:
        assert features in fixed_values
