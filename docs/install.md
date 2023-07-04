# Installation

In BoFire we have several optional depencies.

### Domain and Optimization Algorithms

To install BoFire with optimization tools you can use
```
pip install bofire[optimization]
```
This will also install [BoTorch](https://botorch.org/) that depends on [PyTorch](https://pytorch.org/).
### Design of Experiments

BoFire has functionality to create D-optimal experimental designs via the `doe` module. This module is depends on 
[Cyipopt](https://cyipopt.readthedocs.io/en/stable/). A comfortable way to install Cyipopt and the dependencies is via
```
conda install -c conda-forge cyipopt
```
You have to install Cyipopt manually.
### Just Domain

If you just want a data structure that represents the domain of an optimization problem you can
```
pip install bofire
```

### Cheminformatics

Some features related to molecules and their representation depend on [Rdkit](https://www.rdkit.org/).
```
pip install bofire[optimization,cheminfo]
```

### Development Installation
If you want to [contribute](CONTRIBUTING.md) to BoFire, you might want to install in editable mode including the test dependencies.
After cloning the repository via
```
git clone https://github.com/experimental-design/bofire.git
```
and cd `bofire`, you can proceed with
```
pip install -e .[optimization,cheminfo,docs,tests]
```
