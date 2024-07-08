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
Here, we use the Himmelblau benchmark to demonstrate the ask/tell interface for
proposing new experiments.

```python
from bofire.benchmarks.single import Himmelblau

benchmark = Himmelblau()
samples = benchmark.domain.inputs.sample(10)
experiments = benchmark.f(samples, return_complete=True)

from bofire.data_models.strategies.api import SoboStrategy
from bofire.data_models.acquisition_functions.api import qNEI
import bofire.strategies.api as strategies
sobo_strategy_data_model = SoboStrategy(domain=benchmark.domain, acquisition_function=qNEI())

sobo_strategy = strategies.map(sobo_strategy_data_model)

sobo_strategy.tell(experiments=experiments)
sobo_strategy.ask(candidate_count=1)
```

This gives one step in the optimization loop. We can repeat this many times to
perform Bayesian optimization, exploring the space using intelligent strategies.

## Documentation

Documentation including a section on how to get started can be found under https://experimental-design.github.io/bofire/.

## Contributing

See our [Contributing](./CONTRIBUTING.md) guidelines. If you are not sure about something or find bugs, feel free to create an issue.

By contributing you agree that your contributions will be licensed under the same license as BoFire: [BSD 3-Clause License](./LICENSE).
