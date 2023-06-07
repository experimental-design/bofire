# BoFire

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

## Documentation

Documentation including a section on how to get started can be found under https://experimental-design.github.io/bofire/.

## Contributing

See our [Contributing](./CONTRIBUTING.md) guidelines. If you are not sure about something or find bugs, feel free to create an issue.

By contributing you agree that your contributions will be licensed under the same license as BoFire: [BSD 3-Clause License](./LICENSE).
