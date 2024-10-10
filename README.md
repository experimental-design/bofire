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

## Reference

We would love for you to use BoFire in your work! If you do, please cite [our paper](https://arxiv.org/abs/2408.05040):

    @misc{durholt2024bofire,
      title={BoFire: Bayesian Optimization Framework Intended for Real Experiments},
      author={Johannes P. D{\"{u}}rholt and Thomas S. Asche and Johanna Kleinekorte and Gabriel Mancino-Ball and Benjamin Schiller and Simon Sung and Julian Keupp and Aaron Osburg and Toby Boyne and Ruth Misener and Rosona Eldred and Wagner Steuer Costa and Chrysoula Kappatou and Robert M. Lee and Dominik Linzner and David Walz and Niklas Wulkow and Behrang Shafei},
      year={2024},
      eprint={2408.05040},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2408.05040},
    }

Bayesian optimization in BoFire is based on META's [BoTorch library](https://botorch.org/). For BoTorch, please cite also the [botorch paper](https://proceedings.neurips.cc/paper_files/paper/2020/hash/f5b1b89d98b7286673128a5fb112cb9a-Abstract.html):

    @inproceedings{NEURIPS2020_f5b1b89d,
        author = {Balandat, Maximilian and Karrer, Brian and Jiang, Daniel and Daulton, Samuel and Letham, Ben and Wilson, Andrew G and Bakshy, Eytan},
        booktitle = {Advances in Neural Information Processing Systems},
        editor = {H. Larochelle and M. Ranzato and R. Hadsell and M.F. Balcan and H. Lin},
        pages = {21524--21538},
        publisher = {Curran Associates, Inc.},
        title = {BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization},
        url = {https://proceedings.neurips.cc/paper_files/paper/2020/file/f5b1b89d98b7286673128a5fb112cb9a-Paper.pdf},
        volume = {33},
        year = {2020}
    }

For molecular optimizations, BoFire uses the molecular kernels from the [Gauche library](https://github.com/leojklarner/gauche). If you use the molecular kernels in BoFire please cite also the [gauche paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/f2b1b2e974fa5ea622dd87f22815f423-Abstract-Conference.html):

    @inproceedings{NEURIPS2023_f2b1b2e9,
        author = {Griffiths, Ryan-Rhys and Klarner, Leo and Moss, Henry and Ravuri, Aditya and Truong, Sang and Du, Yuanqi and Stanton, Samuel and Tom, Gary and Rankovic, Bojana and Jamasb, Arian and Deshwal, Aryan and Schwartz, Julius and Tripp, Austin and Kell, Gregory and Frieder, Simon and Bourached, Anthony and Chan, Alex and Moss, Jacob and Guo, Chengzhi and D\"{u}rholt, Johannes Peter and Chaurasia, Saudamini and Park, Ji Won and Strieth-Kalthoff, Felix and Lee, Alpha and Cheng, Bingqing and Aspuru-Guzik, Alan and Schwaller, Philippe and Tang, Jian},
        booktitle = {Advances in Neural Information Processing Systems},
        editor = {A. Oh and T. Naumann and A. Globerson and K. Saenko and M. Hardt and S. Levine},
        pages = {76923--76946},
        publisher = {Curran Associates, Inc.},
        title = {GAUCHE: A Library for Gaussian Processes in Chemistry},
        url = {https://proceedings.neurips.cc/paper_files/paper/2023/file/f2b1b2e974fa5ea622dd87f22815f423-Paper-Conference.pdf},
        volume = {36},
        year = {2023}
    }



## Contributing

See our [Contributing](./CONTRIBUTING.md) guidelines. If you are not sure about something or find bugs, feel free to create an issue.

By contributing you agree that your contributions will be licensed under the same license as BoFire: [BSD 3-Clause License](./LICENSE).
