import json
import os
import pathlib
import sys

pypath = os.path.join(str(pathlib.Path.cwd().parent),"bofire")
sys.path.append(pypath)

from bofire.domain.constraints import (
    LinearEqualityConstraint,
    LinearInequalityConstraint,
)
from bofire.domain.desirability_functions import MaxIdentityDesirabilityFunction
from bofire.domain.domain import Domain
from bofire.domain.features import (
    CategoricalInputFeature,
    ContinuousInputFeature,
    ContinuousOutputFeature,
)
from bofire.strategies.botorch.sobo import BoTorchSoboStrategy as SOBO
from bofire.strategies.strategy import ModelPredictiveStrategy

feature1 = ContinuousInputFeature(key='f21', lower_bound = 0.0, upper_bound=0.7)
feature2 = ContinuousInputFeature(key='f3', lower_bound= 0.0, upper_bound=0.45)
feature3 = ContinuousInputFeature(key='h1', lower_bound = 0.0, upper_bound=0.7)
feature4 = ContinuousInputFeature(key='percent_cta', lower_bound = 0.0, upper_bound=0.02)
feature5 = ContinuousInputFeature(key='percent_ini', lower_bound = 0.002, upper_bound= 0.01)
feature6 = ContinuousInputFeature(key='polymer_content_sample', lower_bound = 0.2, upper_bound=0.6)
feature7 = ContinuousInputFeature(key='t190', lower_bound= 0.0, upper_bound=0.3)
feature8 = ContinuousInputFeature(key='t331', lower_bound = 0.002, upper_bound=0.5)
feature9 = ContinuousInputFeature(key='t432', lower_bound = 0.0, upper_bound=0.5)
feature10 = ContinuousInputFeature(key='t434', lower_bound = 0.0, upper_bound=0.5)

feature11 = CategoricalInputFeature(key='oil', categories=["NB3020", "Berylane", "NB3020/Berylane", "NB3020/Dios"])

#"target": {"type": "min", "steepness": 0.5, "tp": 14.7}
feature_out_1 = ContinuousOutputFeature(key='treat_rate', desirability_function=MaxIdentityDesirabilityFunction(w=1))
# "target": {"type": "identity"}}
feature_out_2 = ContinuousOutputFeature(key='vi_frisch', desirability_function=MaxIdentityDesirabilityFunction(w=1))



input_features = [feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9, feature10]
output_features = [feature_out_1, feature_out_2]

con1 = LinearInequalityConstraint(features = ['t432','t434'], coefficients = [-1,-1], rhs = -0.2)
con2 = LinearEqualityConstraint(features = ["t432", "t434", "t331", "f3", "f21", "h1", "t190"], coefficients = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], rhs=1)

constraints = [con1, con2]

domain = Domain(input_features=input_features, output_features=output_features, constraints=constraints)

from tests.bofire.strategies.dummy import DummyStrategy

strategy = DummyStrategy(domain=domain) #, acquisition_function='QNEI')

print('Ready')
