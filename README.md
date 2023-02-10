
[![Test](https://github.com/experimental-design/bofire/workflows/Tests/badge.svg)](https://github.com/experimental-design/bofire/actions?query=workflow%3ATests)
[![Lint](https://github.com/experimental-design/bofire/workflows/Lint/badge.svg)](https://github.com/experimental-design/bofire/actions?query=workflow%3ALint)
[![Docs](https://github.com/experimental-design/bofire/workflows/Docs/badge.svg)](https://github.com/experimental-design/bofire/actions?query=workflow%3ADocs)
[![PyPI](https://img.shields.io/pypi/v/bofire.svg)](https://pypi.org/project/bofire)
# BoFire
BoFire is a **B**ayesian **O**ptimization **F**ramework **I**ntended for **R**eal **E**xperiments. 

Why BoFire?

BoFire ...

- supports mixed continuous, discrete and categorical parameter spaces for system inputs and outputs,
- separates objectives (minimize, maximize, close-to-target) from the outputs on which they operate,
- supports different specific and generic constraints as well as black-box output constraints,
- provides sampling methods for constrained mixed variable spaces,
- json-serializes problems for use in RESTful APIs and json/bson DBs,
- allows easy out of the box usage of strategies for single and multi-objective Bayesian optimization, and 
- provides a high flexibility on the modelling side if needed.

## Installation

BoFire has BoTorch as its main dependency which depends on PyTorch. In the following you find different options to install BoFire and its dependencies.

### Latest stable release

```
pip install bofire
```

### Current main branch
```
pip install --upgrade git+https://github.com/experimental-design/bofire.git
```

### Development installation
If you want to [contribute](CONTRIBUTING.md) to BoFire, you might want to install in editable mode including the test dependencies.
After cloning the repository via
```
git clone https://github.com/experimental-design/bofire.git
```
and cd `bofire`, you can proceed with
```
pip install -e .[testing]
```
## Documentation

Documentation including a section on how to get started can be found under https://experimental-design.github.io/bofire/.

## Contributing

> TL;DR of the [Contributing](./CONTRIBUTING.md) guidelines.

### Release roadmap

The features for the next release are tracked in the [milestones](https://github.com/experimental-design/bofire/milestones).

### Pull Requests

Pull requests are highly welcome:

1. Create a fork from main.
2. Add or adapt unit tests according to your change.
3. Add doc-strings and update the documentation. You might consider contributing to the tutorials section.
4. Make sure that the GitHub pipeline passes.

### Issues

If you find any issues, Bugs, or not sure about something -  feel free to create an Issue.

## License

By contributing you agree that your contributions will be licensed under the same license as BoFire: [BSD 3-Clause License](./LICENSE).
