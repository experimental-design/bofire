<h1><img src="..\graphics\logos\bofire-long.png" alt="BoFire" style="vertical-align: middle;"> Installation Guide</h1>

### Installation from package index

If you just want a data structure that represents the domain of an optimization problem you can run :

```
pip install bofire
```

### Additional optional dependencies

In BoFire we have several optional dependencies.

- `[optimization]`\: access to optimization tools
- `[cheminfo]`\: includes features related to molecules and their representation
- `[entmoot]`\: enable the ensemble tree model optimization tool
- `[xgb]`\: for extreme gradient boosting
- `[all]`\: install all possible options (except DoE)
- `Design of Experiments (DoE)`\: manual installation necessary

### Domain and Optimization Algorithms

To install BoFire with optimization tools you can use

```
pip install bofire[optimization]
```

This will also install [BoTorch](https://botorch.org/) that depends on [PyTorch](https://pytorch.org/).

### Cheminformatics

Some features related to molecules and their representation depend on [Rdkit](https://www.rdkit.org/).

```
pip install bofire[cheminfo]
```

### Ensemble Tree Model

This option enables tree-based optimization using the Ensemble Tree Model Optimization Tool using tree-based surrogate models and will install [entmoot](https://entmoot.readthedocs.io/) and [lightgbm](https://lightgbm.readthedocs.io/).

```
pip install bofire[entmoot]
```

### Extreme Gradient Boost

For extreme gradient boosting this option will also install [xgboost](https://xgboost.readthedocs.io/).

```
pip install bofire[xgb]
```

### Design of Experiments

BoFire has functionality to create D-optimal experimental designs via the `doe` module. This module is depends on
[Cyipopt](https://cyipopt.readthedocs.io/en/stable/). A comfortable way to install Cyipopt and the dependencies is via

```
conda install -c conda-forge cyipopt
```

You have to install Cyipopt manually.

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
