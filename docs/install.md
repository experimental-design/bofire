# Installation Guide

## Installation from Python Package Index (PyPI)

BoFire can be installed to your Python environment by using `pip`. It can be done by executing

```
pip install bofire
```


!!! tip
    The command from above will install a minimal BoFire version, consisting only of the [data models](data_models_functionals.md). To install BoFire's including its core optimization features, execute:
    ```
    pip install 'bofire[optimization]'
    ```

### Additional optional dependencies

In BoFire, there are several optional dependencies that can be selected during installation via pip, like

```
pip install 'bofire[optimization, cheminfo] # will install bofire with additional dependencies `optimization` and `cheminfo`
```

To get the most our of BoFire, it is recommended to install at least
```
pip install 'bofire[optimization]'
```

The available dependencies are:

- `optimization`: Core Bayesian optimization features.
- `cheminfo`: Cheminformatics utilities.
- `entmoot`: [Entmoot functionality.](https://github.com/cog-imperial/entmoot)
- `xgb`: XGboost surrogates.
- `tests`: Required for running the test suite.
- `docs`: Required for building the documentation.
- `tutorials`: Required for running the [tutorials.](https://github.com/experimental-design/bofire/tree/main/tutorials)
- `all`: Install all possible options (except DoE)

!!! warning
    BoFire has the functionalities for creating D, E, A, G, K and I-optimal experimental designs via the `DoEStrategy`. This feature depends on [cyipopt](https://cyipopt.readthedocs.io/en/stable/) which is a python interface to `ipopt`. Unfortunately, it is not possible to install `cyipopt` including `ipopt` via pip. A solution is to install `cyipopt` and its dependencies via conda:

    ```
    conda install -c conda-forge cyipopt
    ```

    We are working on a solution that makes BoFire's model based DoE functionalities also accessible to users which do not have `cyipopt` available.


## Development Installation

If you want to [contribute](CONTRIBUTING.md) to BoFire, it is recommended to install the repository in editable mode (`-e`).

After cloning the repository via
```
git clone https://github.com/experimental-design/bofire.git
```
and navigating to the repositories root folder (`cd bofire`), you can proceed with
```
pip install -e ".[optimization, tests]" # include optional dependencies as you wish
```
