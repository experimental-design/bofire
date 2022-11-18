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
from bofire.domain.domain import Domain
from bofire.domain.features import CategoricalInput, ContinuousInput, ContinuousOutput
from bofire.domain.objectives import MaximizeObjective
from bofire.strategies.botorch.sobo import BoTorchSoboStrategy as SOBO
from bofire.strategies.strategy import PredictiveStrategy

feature1 = ContinuousInput(key='f21', lower_bound = 0.0, upper_bound=0.7)
feature2 = ContinuousInput(key='f3', lower_bound= 0.0, upper_bound=0.45)
feature3 = ContinuousInput(key='h1', lower_bound = 0.0, upper_bound=0.7)
feature4 = ContinuousInput(key='percent_cta', lower_bound = 0.0, upper_bound=0.02)
feature5 = ContinuousInput(key='percent_ini', lower_bound = 0.002, upper_bound= 0.01)
feature6 = ContinuousInput(key='polymer_content_sample', lower_bound = 0.2, upper_bound=0.6)
feature7 = ContinuousInput(key='t190', lower_bound= 0.0, upper_bound=0.3)
feature8 = ContinuousInput(key='t331', lower_bound = 0.002, upper_bound=0.5)
feature9 = ContinuousInput(key='t432', lower_bound = 0.0, upper_bound=0.5)
feature10 = ContinuousInput(key='t434', lower_bound = 0.0, upper_bound=0.5)

feature11 = CategoricalInput(key='oil', categories=["NB3020", "Berylane", "NB3020/Berylane", "NB3020/Dios"])

#"target": {"type": "min", "steepness": 0.5, "tp": 14.7}
feature_out_1 = ContinuousOutput(key='treat_rate', objective=MaximizeObjective(w=1))
# "target": {"type": "identity"}}
feature_out_2 = ContinuousOutput(key='vi_frisch', objective=MaximizeObjective(w=1))



input_features = [feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9, feature10]
output_features = [feature_out_1, feature_out_2]

con1 = LinearInequalityConstraint(features = ['t432','t434'], coefficients = [-1,-1], rhs = -0.2)
con2 = LinearEqualityConstraint(features = ["t432", "t434", "t331", "f3", "f21", "h1", "t190"], coefficients = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], rhs=1)

constraints = [con1, con2]

domain = Domain(input_features=input_features, output_features=output_features, constraints=constraints)

strategy = SOBO(domain=domain, acquisition_function='QNEI')

from tests.bofire.strategies.dummy import DummyStrategy

strategy = DummyStrategy(domain=domain) #, acquisition_function='QNEI')



print('Ready')
