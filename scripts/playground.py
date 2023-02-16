import os
import pathlib
import sys

pypath = os.path.join(str(pathlib.Path.cwd().parent), "bofire")
sys.path.append(pypath)

from bofire.domain.constraint import (  # noqa: E402
    LinearEqualityConstraint,
    LinearInequalityConstraint,
)
from bofire.domain.domain import Domain  # noqa: E402
from bofire.domain.feature import (  # noqa: E402
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
)
from bofire.domain.objectives import MaximizeObjective  # noqa: E402
from bofire.strategies.botorch.qehvi import BoTorchQehviStrategy  # noqa: E402
from tests.bofire.domain.test_domain_validators import (  # noqa: E402
    generate_experiments,
)
from tests.bofire.strategies.botorch.test_qehvi import (  # noqa: E402
    BOTORCH_QEHVI_STRATEGY_SPECS,
)

feature1 = ContinuousInput(key="x1", lower_bound=0.0, upper_bound=0.7)
feature2 = ContinuousInput(key="x2", lower_bound=0.0, upper_bound=0.45)
feature3 = ContinuousInput(key="x3", lower_bound=0.0, upper_bound=0.7)

feature4 = CategoricalInput(key="c1", categories=["A", "B", "C", "D"])

# "target": {"type": "min", "steepness": 0.5, "tp": 14.7}
feature_out_1 = ContinuousOutput(key="y1", objective=MaximizeObjective(w=1))
# "target": {"type": "identity"}}
feature_out_2 = ContinuousOutput(key="y2", objective=MaximizeObjective(w=1))


input_features = [feature1, feature2, feature3, feature4]
output_features = [feature_out_1, feature_out_2]

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

strategy = BoTorchQehviStrategy(**BOTORCH_QEHVI_STRATEGY_SPECS["valids"][2])

experiments_train = generate_experiments(domain, 10)
experiments_test = generate_experiments(domain, 10)
strategy.tell(experiments_train)
candidates = strategy._choose_from_pool(experiments_test, 5)

strategy.get_fbest(experiments_test)

strategy = BoTorchQehviStrategy(
    domain=domain
)  # #ref_point = {'x1': 1., 'x2': 4., 'x3': 6.}, acquisition_function='QNEI')

# strategy = BoTorchQparegoStrategy(domain=domain, ref_point = {'x1': 1., 'x2': 4., 'x3': 6.}, acquisition_function='QEI', model_specs=[model_specs])
# strategy = DummyStrategy(domain=domain) #, acquisition_function='QNEI')


print("Ready")
