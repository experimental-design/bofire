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
- `MoboStrategy`: Bayesian optimization strategy that optimizes a hypervolume based acquisition function for pareto-pased multi-objective optimization.
- `QparegoStrategy`: Parallel Ego strat
- `MultifidelityStrategy:`
- `EntingStrategy`: Strategy based on the `Entmoot` package that uses tree-based surrogate models to perform both single-objective and multiobjective optimization.

## Combining Strategies

In BoFire, the `StepwiseStrategy` operates on a sequence of strategies and determines when to switch between them based on customizable logical operators.

The `StepwiseStrategy` is comprised of a sequence of `Step`s, where each `Step` consists of the following three attributes:

- `strategy_data`: data model of the strategy which should be executed in this step.
- `condition`: A logical expression that determines when this `step`'s strategy should be executed. The `StepwiseStrategy` evaluates each step in order and selects the first strategy whose condition evaluates to `True`.
- `transform`:

``` python

from bofire.data_models.strategies.api import StepwiseStrategy, SoboStrategy, RandomStrategy,


```


different strategies can be combined to a sequence of strategies.
