import random

import numpy as np
import pandas as pd
import pytest

import bofire.data_models
import tests.bofire.data_models.specs.api as specs
from bofire.data_models.api import Domain, Inputs
from bofire.data_models.constraints.api import (
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
)
from bofire.data_models.features.api import CategoricalInput, ContinuousInput
from tests.bofire.data_models.test_samplers import test_PolytopeSampler
