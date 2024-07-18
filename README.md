<a href=https://experimental-design.github.io/bofire/>
  <img width="350" src="https://raw.githubusercontent.com/experimental-design/bofire/main/graphics/logos/bofire-long.png" alt="BoFire Logo" />
</a>

<hr/>

[![Test](https://github.com/experimental-design/bofire/workflows/Tests/badge.svg)](https://github.com/experimental-design/bofire/actions?query=workflow%3ATests)
[![Lint](https://github.com/experimental-design/bofire/workflows/Lint/badge.svg)](https://github.com/experimental-design/bofire/actions?query=workflow%3ALint)
[![Docs](https://github.com/experimental-design/bofire/workflows/Docs/badge.svg)](https://github.com/experimental-design/bofire/actions?query=workflow%3ADocs)
[![PyPI](https://img.shields.io/pypi/v/bofire.svg)](https://pypi.org/project/bofire)

BoFire is a **B**ayesian **O**ptimization **F**ramework **I**ntended for **R**eal **E**xperiments. 

Why BoFire?

BoFire ...

- supports mixed continuous, discrete and categorical parameter spaces for system inputs and outputs,
- separates objectives (minimize, maximize, close-to-target) from the outputs on which they operate,
- supports different specific and generic constraints as well as black-box output constraints,
- can provide flexible DoEs that fulfill constraints,
- provides sampling methods for constrained mixed variable spaces,
- serializes problems for use in RESTful APIs and json/bson DBs,
- allows easy out of the box usage of strategies for single and multi-objective Bayesian optimization, and 
- provides a high flexibility on the modelling side if needed.

## Installation

In our [docs](https://experimental-design.github.io/bofire/install/),
you can find all different options for the BoFire installation.
To install all BoFire-features you need to run
```
pip install bofire[optimization,cheminfo]
```
This will also install [BoTorch](https://botorch.org/) that depends on 
[PyTorch](https://pytorch.org/). To use the DoE package, you need to install
[Cyipopt](https://cyipopt.readthedocs.io/en/stable/)
additionally, e.g., via
```
conda install -c conda-forge cyipopt
```

## Getting Started
For a more complete introduction to BoFire, please look at the [Getting Started documentation](https://experimental-design.github.io/bofire/getting_started/).

We first must define the domain of the optimization problem.

```python
from bofire.data_models.features.api import ContinuousInput, ContinuousOutput
from bofire.data_models.objectives.api import MaximizeObjective
from bofire.data_models.constraints.api import NChooseKConstraint
from bofire.data_models.domain.api import Domain, Inputs, Outputs, Constraints

input_features = Inputs(features=[
    ContinuousInput(key="x1", bounds=(0,1)),
    ContinuousInput(key="x2", bounds=(0,1)),
    ContinuousInput(key="x3", bounds=(0,1)),
])

output_features = Outputs(features=[
    ContinuousOutput(key="y", objective=MaximizeObjective())
])

constraints = Constraints(constraints=[
    NChooseKConstraint(
        features=["x1", "x2", "x3"],
        min_count=1, max_count=2, none_also_valid=False)
])

domain = Domain(
    inputs=input_features, 
    outputs=output_features, 
    constraints=constraints
)
```

You can also use one of the many benchmarks available in BoFire.
Here, we use the Detergent benchmark to demonstrate the ask/tell interface for
proposing new experiments with multi-objective Bayesian optimization.

```python
import bofire.strategies.api as strategies
from bofire.benchmarks.api import Detergent
from bofire.data_models.strategies.api import QnehviStrategy, RandomStrategy

# create benchmark
detergent = Detergent()
domain = detergent.domain

# create initial data with the random strategy while satisfying constraints
sampler = strategies.map(RandomStrategy(domain=domain))
initial_samples = sampler.ask(2)
experiments = detergent.f(initial_samples, return_complete=True)

# Bayesian optimization
mobo_strategy = strategies.map(QnehviStrategy(domain=domain))
n_experiments = 4
for _ in range(n_experiments):
    mobo_strategy.tell(experiments=experiments)
    candidates = mobo_strategy.ask(candidate_count=1)
    experiments = detergent.f(candidates, return_complete=True)

# Print all told experiments
print(mobo_strategy.experiments)
```

## Documentation

Documentation including a section on how to get started can be found under https://experimental-design.github.io/bofire/.

## Contributing

See our [Contributing](./CONTRIBUTING.md) guidelines. If you are not sure about something or find bugs, feel free to create an issue.

By contributing you agree that your contributions will be licensed under the same license as BoFire: [BSD 3-Clause License](./LICENSE).
