# Strategies

Strategies are the key ingredient of BoFire that explore the search space defined in the `Domain` and provide candidates for the next experiment or batch of experiments. Available strategies can be clustered in the following subclasses.

## Non-predictive Strategies

BoFire offers the following strategies for sampling from the search space (no output features need to be provided in the `Domain`):

- `RandomStrategy`: This strategy proposes candidates by (quasi-)random sampling from the search space. It is applicable to almost all combinations of input features and constraints.

- `DoEStrategy`: This strategy offers model-based DoE approaches for proposing candidates.

- `FractionalFactorialStrategy`: This strategy should be used to generate (fractional-)factorial designs.

## Predictive Strategies

Predictive strategies are making use of (Bayesian) surrogate models to provide candidates with the intention to achieve certain goals depending on the provided objectives of the output features.

The following predictive strategies are available:

- `SoboStrategy`: Bayesian optimization strategy that optimizes a single-objective acquisition function. For multi-objective domains, different scalarizations are possible as implemented in the `AdditiveSoboStrategy`, `MultiplicativeSoboStrategy`, `AdditiveMultiplicativeSoboStrategy` and `CustomSoboStrategy`.
- `MoboStrategy`: Bayesian optimization strategy that optimizes a hypervolume based acquisition function for pareto-based multi-objective optimization.
- `QparegoStrategy`: Parallel ParEGO strategy for multiobjective optimization.
- `MultifidelityStrategy:` Single objective multi-fidelity BO as described [here](https://www.sciencedirect.com/science/article/pii/S0098135423000637)
- `EntingStrategy`: Strategy based on the `Entmoot` [package](https://github.com/cog-imperial/entmoot) that uses tree-based surrogate models to perform both single-objective and multiobjective optimization.

## Combining Strategies

In BoFire, the `StepwiseStrategy` operates on a sequence of strategies and determines when to switch between them based on customizable logical operators.

The `StepwiseStrategy` is comprised of a sequence of `Step`s, where each `Step` consists of the following three attributes:

- `strategy_data`: data model of the strategy which should be executed in this step.
- `condition`: A logical expression that determines when this `step`'s strategy should be executed. The `StepwiseStrategy` evaluates each step in order and selects the first strategy whose condition evaluates to `True`.
- `transform`: An object that can be used to transform experiments and/or candidates before they enter/leave the strategy assigned in the step.

The following example demonstrates how to combine a `RandomStrategy` with a `SoboStrategy` using the `StepwiseStrategy`. In this setup, the `RandomStrategy` is applied initially to propose candidates until 10 experiments have been completed. Once this threshold is reached, the strategy automatically switches to the `SoboStrategy` for subsequent candidate generation.

``` python

import bofire.strategies.api as strategies
from bofire.benchmarks.api import Himmelblau
from bofire.data_models.strategies.api import (
    AlwaysTrueCondition,
    NumberOfExperimentsCondition,
    RandomStrategy,
    SoboStrategy,
    Step,
    StepwiseStrategy,
)


domain = Himmelblau().domain

strategy_data = StepwiseStrategy(
    domain=domain,
    steps=[
        Step(
            strategy_data=RandomStrategy(domain=domain),
            condition=NumberOfExperimentsCondition(n_experiments=10),
        ),
        Step(
            strategy_data=SoboStrategy(domain=domain), condition=AlwaysTrueCondition()
        ),
    ],
)

strategy = strategies.map(strategy_data)
```

When dealing with output constraints, it is often beneficial to ensure that a certain number of experiments satisfy these constraints before proceeding with the main optimization. If no feasible experiments have been found, the strategy should prioritize generating candidates likely to fulfill the output constraints. This can be accomplished by using the `qLogPF` (Probability of Feasibility) acquisition function together with the `FeasibleExperimentCondition`. This approach allows the optimization process to focus first on feasibility, and only switch to the main objective once enough feasible experiments are available.

In the following example, a `RandomStrategy` is applied for the initial 10 experiments to broadly explore the search space. Afterward, the `SoboStrategy` with the `qLogPF` acquisition function is used to prioritize finding at least one feasible experiment that satisfies the output constraints. Once this feasibility criterion is met, the strategy transitions to the standard `SoboStrategy` to focus on optimizing the main objective.

```python
import bofire.strategies.api as strategies
from bofire.benchmarks.api import DTLZ2
from bofire.data_models.acquisition_functions.api import qLogPF
from bofire.data_models.objectives.api import MaximizeSigmoidObjective
from bofire.data_models.strategies.api import (
    AlwaysTrueCondition,
    FeasibleExperimentCondition,
    NumberOfExperimentsCondition,
    RandomStrategy,
    SoboStrategy,
    Step,
    StepwiseStrategy,
)


# create a domain with one output constraint by assigning a MaximizeSigmoidObjective
# to the output with key "f_1"
domain = DTLZ2(dim=6).domain
domain.outputs.get_by_key("f_1").objective = MaximizeSigmoidObjective(
    tp=0.5, steepness=100
)


strategy_data = StepwiseStrategy(
    domain=domain,
    steps=[
        Step(
            strategy_data=RandomStrategy(domain=domain),
            condition=NumberOfExperimentsCondition(n_experiments=10),
        ),
        Step(
            strategy_data=SoboStrategy(domain=domain, acquisition_function=qLogPF()),
            condition=FeasibleExperimentCondition(n_required_feasible_experiments=1),
        ),
        Step(
            strategy_data=SoboStrategy(domain=domain), condition=AlwaysTrueCondition()
        ),
    ],
)

strategy = strategies.map(strategy_data)

```
