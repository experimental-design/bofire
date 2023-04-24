# Getting started

In BoFire, an optimization problem is defined by defining a domain containing input and output features as well as constraints (optional). 

<!-- 
TODO:

- [ ] Update after Notebooks fixed
- [ ] Add APIs
- [ ] Add Data models
- [ ] 
 -->

## Features

Input features can be continuous, discrete, categorical, or categorical with descriptors:

```python
from bofire.domain.features import ContinuousInput, DiscreteInput, CategoricalInput, CategoricalDescriptorInput

x1 = ContinuousInput(key="x1", lower_bound=0, upper_bound=1)
x2 = ContinuousInput(key="x2", lower_bound=0, upper_bound=1)
x3 = ContinuousInput(key="x3", lower_bound=0, upper_bound=1)
x4 = DiscreteInput(key="x4", values=[1, 2, 5, 7.5])
x5 = CategoricalInput(key="x5", categories=["A", "B", "C"], allowed=[True,True,False])
x6 = CategoricalDescriptorInput(key="x6", categories=["c1", "c2", "c3"], descriptors=["d1", "d2"], values = [[1,2],[2,5],[1,7]])
```

As output features, currently only continuous output features are supported. Each output feature has to have an objective, which can be a minimize or maximize objective. Furthermore, we can define weights between 0 and 1 in case the objectives should not be weighted equally.

```python
from bofire.domain.features import ContinuousOutput
from bofire.domain.objectives import MaximizeObjective, MinimizeObjective

objective1 = MaximizeObjective(
    w=1.0, 
    lower_bound=0, 
    upper_bound=1,
)
y1 = ContinuousOutput(key="y1", objective=objective1)

objective2 = MinimizeObjective(
    w=1.0
)
y2 = ContinuousOutput(key="y2", objective=objective2)
```
In- and output features are collected in respective feature lists.

```python
from bofire.domain.features import InputFeatures, OutputFeatures

input_features = InputFeatures(features = [x1, x2, x3, x4, x5, x6])
output_features = OutputFeatures(features=[y1, y2])
```

Individual features can be retrieved by name.

```python
x5 = input_features.get_by_key('x5')
>>> CategoricalInput(key='x5', type='CategoricalInput', categories=['A', 'B', 'C'], allowed=[True, True, False])
```

All feature keys of the input or output features can be returned by the get_keys() method.

```python
input_features.get_keys()
>>> ["x1", "x2", "x3", "x4", "x5"]

output_features.get_keys()
>>> ["y1", "y2"]
```

The input feature container further provides methods to return a feature container with only all fixed or all free features.

```python
free_inputs = input_features.get_free()
fixed_inputs = input_features.get_fixed()
```

We can sample from individual input features or input feature containers:

```python
samples_x5 = x5.sample(2)
samples_x5

>>> 0    B
1    B
Name: x5, dtype: object

X = input_features.sample(n=5)
print(X)
>>>      x1        x2        x3   x4 x5
0  0.760116  0.063584  0.518885  7.5  A
1  0.807928  0.496213  0.885545  1.0  C
2  0.351253  0.993993  0.340414  5.0  B
3  0.385825  0.857306  0.355267  1.0  C
4  0.191907  0.993494  0.384322  2.0  A
```


## Constraints

The search space can be further defined by constraints on the input features. BoFire supports linear equality and inequality constraints, as well as non-linear equality and inequality constraints.

### Linear constraints

`LinearEqualityConstraint` and `LinearInequalityConstraint` are expressions of the form $\sum_i a_i x_i = b$ or $\leq b$ for equality and inequality constraints respectively.
They take a list of names of the input features they are operating on, a list of left-hand-side coefficients $a_i$ and a right-hand-side constant $b$.

<!-- TODO: Check if LIC also allows x > y -->

```python
from bofire.domain.constraints import LinearEqualityConstraint, LinearInequalityConstraint

# A mixture: x1 + x2 + x3 = 1
constr1 = LinearEqualityConstraint(features=["x1", "x2", "x3"], coefficients=[1,1,1], rhs=1)

# x1 + 2 * x3 < 0.8
constr2 = LinearInequalityConstraint(features=["x1", "x3"], coefficients=[1, 2], rhs=0.8)
```
Because of the product $a_i x_i$, linear constraints cannot operate on categorical parameters.

### Nonlinear constraints

`NonlinearEqualityConstraint` and `NonlinearInequalityConstraint` take any expression that can be evaluated by [pandas.eval](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.eval.html), including mathematical operators such as `sin`, `exp`, `log10` or exponentiation.

```python
from bofire.domain.constraints import NonlinearEqualityConstraint, NonlinearInequalityConstraint

# The unit circle: x1**2 + x2**2 = 1
constr3 = NonlinearEqualityConstraint(expression="x1**2 + x2**2 - 1")
```
Nonlinear constraints can also operate on categorical parameters and support conditional statements.

```python
# Require x1 < 0.5 if x5 == "A"
constr4 = NonlinearInequalityConstraint(expression="(x1 - 0.5) * (x5 =='A')")
```

<!-- TODO: Change NchooseK once (0, min, max) available -->

### Combinatorial constraint

Use `NChooseKConstraint` to express that we only want to have $k$ out of the $n$ parameters to take positive values.
Think of a mixture, where we have long list of possible ingredients, but want to limit number of ingredients in any given recipe.

```python
from bofire.domain.constraints import NChooseKConstraint

# Only 2 or 3 out of 3 parameters can be greater than zero
constr5 = NChooseKConstraint(features=["x1", "x2", "x3"], min_count=2, max_count=3, none_also_valid=True)
```
Note that we have to set a boolean, if None is also a valid selection, e.g. if we want to have 2 or 3 or none of the ingredients in our recipe.

Similar to the features, constraints can be grouped in a container which acts as the union constraints.
```python
from bofire.domain.constraints import Constraints

constraints = Constraints(constraints=[constr1, constr2])
```

We can check whether a point satisfies individual constraints or the list of constraints.
```python
constr2.is_fulfilled(X).values
>>> array([False, False, True, True, True])
```

Output features are not constrained by constraints but via sigmoid-shaped objectives passed as argument to the respective feature. 

```python
from bofire.domain.objectives import MinimizeSigmoidObjective

output_constraint = MinimizeSigmoidObjective(
    w=1.0, 
    steepness=10,
    tp=0.5
)
y3= ContinuousOutput(key="y3", objective=output_constraint)

output_features = OutputFeatures(features=[y1, y2, y3])
```

The shape of the output features can be plotted:

```python
_ = y3.plot(lower=0, upper=1)
```

## The domain

Finally, the domain can be instantiated:

```python
from bofire.domain.domain import Domain

domain = Domain(
    input_features=input_features, 
    output_features=output_features, 
    constraints=constraints
    )
```

A summary of the defined features and constraints can be obtained by the methods `get_feature_reps_df()` and `get_constraint_reps_df()`:

```python
domain.get_feature_reps_df()
domain.get_constraint_reps_df()
```


## Set up a strategy

To solve the optimization problem, we further need a solving strategy. BoFire supports strategies without a prediction model such as a random strategy and predictive strategies which are based on a prediction model.

All strategies contain an ask method returning a defined number of candidate experiments. 

### Random strategy

```python
from bofire.strategies.random import RandomStrategy

random_strategy = RandomStrategy(domain=domain)
random_candidates = random_strategy.ask(2)

random_candidates
>>> 	x1	x2	x3	x4	x6	x5
0	0.646864	0.317559	0.035577	1.0	c1	B
1	0.341934	0.525235	0.132830	2.0	c3	A
```

### Single objective Bayesian Optimization strategy

Since a predictive strategy includes a prediction model, we need to generate some historical data, which we can afterwards pass as training data to the strategy via the tell method.

```python
import pandas as pd

X = pd.DataFrame(
    data={
        "x1": [0, 0.5, 0.2, 0.1, 0, 0.5, 0.2, 0.1, 0, 0.5, 0.2, 0.1],
        "x2": [0.7, 0.5, 0.6, 0.7, 0.7, 0.5, 0.6, 0.7, 0.7, 0.5, 0.6, 0.7],
        "x3": [0.3, 0, 0.2, 0.2, 0.3, 0, 0.2, 0.2, 0.3, 0, 0.2, 0.2],
        "x4": [1, 2, 5, 7.5, 2, 5, 7.5, 1, 5, 7.5, 1, 2],
        "x5": ["C", "B", "A", "A", "B", "B", "A", "B", "A", "A", "B", "A"],
        "x6": ["c1", "c2", "c1", "c2", "c3", "c3", "c1", "c2", "c3", "c1", "c2", "c3"],
        "y1": [2, 4.25, 6.16, 9.59, 5, 8.25, 8.66, 3.09, 8, 8.75, 3.16, 5.09],
        "valid_y1": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "y2": [3, 2, 1, 1, 2, 2, 1, 2, 1, 1, 2, 1],
        "valid_y2": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "y3": [0.6, 0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.7, 0.8, 1, 0.2],
        "valid_y3": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    }
)
```
Note that we defined one entry of x5 to be the non-allowed category. If we would not pass historical data containing the non-allowed category, BoFire would suggest to remove the unused category from the domain:

```python
sobo_strategy.tell(X.loc[1:])
```

```python
sobo_strategy.tell(X, replace=True)
sobo_strategy.ask(candidate_count=2, add_pending=True)
```
The argument replace = True  determines, if former experiments stored to the domain should be overwritten or the new experiments should be append.