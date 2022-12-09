import json
import os
import pathlib
import sys

pypath = os.path.join(str(pathlib.Path.cwd().parent), "bofire")
sys.path.append(pypath)

import torch

from bofire.domain.constraints import (
    LinearEqualityConstraint,
    LinearInequalityConstraint,
)
from bofire.domain.domain import Domain
from bofire.domain.features import CategoricalInput, ContinuousInput, ContinuousOutput
from bofire.domain.objectives import (
    ConstantObjective,
    MaximizeObjective,
    MaximizeSigmoidObjective,
    MinimizeObjective,
)
from bofire.strategies.botorch.base import ModelSpec
from bofire.strategies.botorch.qehvi import BoTorchQehviStrategy, BoTorchQnehviStrategy
from bofire.strategies.botorch.qparego import BoTorchQparegoStrategy
from bofire.strategies.botorch.sobo import BoTorchSoboStrategy as SOBO
from bofire.strategies.botorch.utils.objectives import (
    AdditiveObjective,
    MultiplicativeObjective,
)
from bofire.strategies.strategy import PredictiveStrategy
from tests.bofire.domain.test_domain_validators import (
    generate_candidates,
    generate_experiments,
)
from tests.bofire.strategies.botorch.test_qehvi import BOTORCH_QEHVI_STRATEGY_SPECS
from tests.bofire.strategies.dummy import DummyStrategy
from tests.bofire.utils.test_multiobjective import dfs, invalid_domains

feature1 = ContinuousInput(key="x1", lower_bound=0.0, upper_bound=0.7)
feature2 = ContinuousInput(key="x2", lower_bound=0.0, upper_bound=0.45)
feature3 = ContinuousInput(key="x3", lower_bound=0.0, upper_bound=0.7)

feature4 = CategoricalInput(key="c1", categories=["A", "B", "C", "D"])


feature_out_1 = ContinuousOutput(
    key="y1", objective=MaximizeObjective(w=1)  # , steepness=10.0, tp=0.5)
)

feature_out_2 = ContinuousOutput(
    key="y2", objective=MaximizeSigmoidObjective(w=1, steepness=10.0, tp=2)
)

feature_out_3 = ContinuousOutput(
    key="y3", objective=MaximizeObjective(w=1)  # , steepness=10.0, tp=0.5)
)

feature_out_4 = ContinuousOutput(key="y4", objective=None)

input_features = [feature1, feature2, feature3, feature4]
output_features = [feature_out_1, feature_out_2, feature_out_3, feature_out_4]

con1 = LinearInequalityConstraint(
    features=["x1", "x2"], coefficients=[-1, -1], rhs=-0.2
)
con2 = LinearEqualityConstraint(
    features=["x1", "x2", "x3"], coefficients=[1.0, 1.0, 1.0], rhs=1
)

constraints = [con1, con2]

domain = Domain(
    input_features=input_features,
    output_features=output_features,
    constraints=constraints,
)

# strategy = SOBO(domain=domain, acquisition_function="QNEI")

strategy = BoTorchQehviStrategy(domain=domain)

# strategy = BoTorchQparegoStrategy(
#     domain=domain,
#     categorical_method="FREE",
#     descriptor_method="FREE",
#     categorical_encoding="ONE_HOT",
# )

experiments_train = generate_experiments(domain, 10)
experiments_test = generate_experiments(domain, 10)
strategy.tell(experiments_train)
# candidates = strategy._choose_from_pool(experiments_test, 5)
candidates2 = strategy.ask(candidate_count=2)

# model_specs = ModelSpec(output_feature = 'y1', input_features = ['x1', 'x2', 'x3'], kernel = "MATERN_25", ard =True, scaler = "NORMALIZE")
# strategy = BoTorchQparegoStrategy(domain=domain, ref_point = {'x1': 1., 'x2': 4., 'x3': 6.}, acquisition_function='QEI', model_specs=[model_specs])
# strategy = DummyStrategy(domain=domain) #, acquisition_function='QNEI')


print("Ready")
