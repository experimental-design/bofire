# Getting started

In BoFire, an optimization problem is defined by defining a domain containing input and output features as well as constraints (optional). 

<!-- 
TODO:

- [x] Update after Notebooks fixed
- [ ] Add APIs
- [ ] Add Data models
- [ ] 
 -->

## Features

Input features can be continuous, discrete, categorical, or categorical with descriptors:

```python
from bofire.data_models.features.api import ContinuousInput, DiscreteInput, CategoricalInput, CategoricalDescriptorInput

x1 = ContinuousInput(key="x1", lower_bound=0, upper_bound=1)
x2 = ContinuousInput(key="x2", lower_bound=0, upper_bound=1)
x3 = ContinuousInput(key="x3", lower_bound=0, upper_bound=1)
x4 = DiscreteInput(key="x4", values=[1, 2, 5, 7.5])
x5 = CategoricalInput(key="x5", categories=["A", "B", "C"], allowed=[True,True,False])
x6 = CategoricalDescriptorInput(key="x6", categories=["c1", "c2", "c3"], descriptors=["d1", "d2"], values = [[1,2],[2,5],[1,7]])
```

As output features, currently only continuous output features are supported. Each output feature has to have an objective, which can be a minimize or maximize objective. Furthermore, we can define weights between 0 and 1 in case the objectives should not be weighted equally.

```python
from bofire.data_models.features.api import ContinuousOutput
from bofire.data_models.objectives.api import MaximizeObjective, MinimizeObjective

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
from bofire.data_models.domain.api import Inputs, Outputs

input_features = Inputs(features = [x1, x2, x3, x4, x5, x6])
output_features = Outputs(features=[y1, y2])
```

Individual features can be retrieved by name.

```python
x5 = input_features.get_by_key('x5')
x5
>>> CategoricalInput(type='CategoricalInput', key='x5', categories=['A', 'B', 'C'], allowed=[True, True, False])

```

All feature keys of the input or output features can be returned by the get_keys() method.

```python
input_features.get_keys()
>>> ["x1", "x2", "x3", "x4", "x6", "x5"]

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

>>> 0    A
1    B
Name: x5, dtype: object

X = input_features.sample(n=10)
print(X)
>>>      x1	          x2	      x3	 x4	x6	x5
0	0.446951	0.233108	0.807302	1.0	c2	 B
1	0.902728	0.905087	0.299669	5.0	c1	 B
2	0.437934	0.846569	0.023396	7.5	c1	 A
3	0.343520	0.058300	0.031096	5.0	c3	 A
4	0.204773	0.211742	0.103538	5.0	c1	 B
5	0.936187	0.330185	0.903871	5.0	c1	 A
6	0.426863	0.048808	0.187039	7.5	c1	 A
7	0.361787	0.982932	0.287825	5.0	c3	 A
8	0.517546	0.210103	0.531776	5.0	c1	 B
9	0.602998	0.046365	0.683659	5.0	c1	 B
```


## Constraints

The search space can be further defined by constraints on the input features. BoFire supports linear equality and inequality constraints, as well as non-linear equality and inequality constraints.

### Linear constraints

`LinearEqualityConstraint` and `LinearInequalityConstraint` are expressions of the form $\sum_i a_i x_i = b$ or $\leq b$ for equality and inequality constraints respectively.
They take a list of names of the input features they are operating on, a list of left-hand-side coefficients $a_i$ and a right-hand-side constant $b$.

<!-- TODO: Check if LIC also allows x > y -->

```python
from bofire.data_models.constraints.api import LinearEqualityConstraint, LinearInequalityConstraint

# A mixture: x1 + x2 + x3 = 1
constr1 = LinearEqualityConstraint(features=["x1", "x2", "x3"], coefficients=[1,1,1], rhs=1)

# x1 + 2 * x3 < 0.8
constr2 = LinearInequalityConstraint(features=["x1", "x3"], coefficients=[1, 2], rhs=0.8)
```
Because of the product $a_i x_i$, linear constraints cannot operate on categorical parameters.

### Nonlinear constraints

`NonlinearEqualityConstraint` and `NonlinearInequalityConstraint` take any expression that can be evaluated by [pandas.eval](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.eval.html), including mathematical operators such as `sin`, `exp`, `log10` or exponentiation.

```python
from bofire.data_models.constraints.api import NonlinearEqualityConstraint, NonlinearInequalityConstraint

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
from bofire.data_models.constraints.api import NChooseKConstraint

# Only 2 or 3 out of 3 parameters can be greater than zero
constr5 = NChooseKConstraint(features=["x1", "x2", "x3"], min_count=2, max_count=3, none_also_valid=True)
```

Note that we have to set a boolean, if None is also a valid selection, e.g. if we want to have 2 or 3 or none of the ingredients in our recipe.

Similar to the features, constraints can be grouped in a container which acts as the union constraints.

```python
from bofire.data_models.domain.api import Constraints

constraints = Constraints(constraints=[constr1, constr2])
```

We can check whether a point satisfies individual constraints or the list of constraints.
```python
constr2.is_fulfilled(X).values
>>> array([False, False,  True,  True,  True, False, False, False, False,
       False])
```

Output features are not constrained by constraints but via sigmoid-shaped objectives passed as argument to the respective feature. 

```python
from bofire.data_models.objectives.api import MinimizeSigmoidObjective

output_constraint = MinimizeSigmoidObjective(
    w=1.0, 
    steepness=10,
    tp=0.5
)
y3= ContinuousOutput(key="y3", objective=output_constraint)

output_features = Outputs(features=[y1, y2, y3])
```

The shape of the output features can be plotted:

```python
_ = y3.plot(lower=0, upper=1)
```

## The domain

Finally, the domain can be instantiated:

```python
from bofire.data_models.domain.api import Domain

domain = Domain(
    inputs=input_features, 
    outputs=output_features, 
    constraints=constraints
    )
```

<!-- Await Behrangs PR for functionality -->

Alternatively, a single objective Domain may also be instantiated, for example from a list:

```python
domain_single_objective = Domain.from_list(
    inputs=[x1, x2, x3, x4, x5, x6], 
    outputs=[y1], 
    constraints=[]
    )
```

A summary of the defined features and constraints can be obtained by the methods `get_feature_reps_df()` and `get_constraint_reps_df()`:

```python
domain.get_feature_reps_df()
>>>            Type	    Description
x1	ContinuousInput	            [0.0,1.0]
x2	ContinuousInput	            [0.0,1.0]
x3	ContinuousInput	            [0.0,1.0]
x4	DiscreteInput	            type='DiscreteInput' key='x4' unit=None values...
x6	CategoricalDescriptorInput	3 categories
x5	CategoricalInput	        3 categories
y1	ContinuousOutput	        ContinuousOutputFeature
y2	ContinuousOutput	        ContinuousOutputFeature
y3	ContinuousOutput	        ContinuousOutputFeature
```

```python
domain.get_constraint_reps_df()
>>>                     Type	Description
0	LinearEqualityConstraint	1.0 * x1 + 1.0 * x2 + 1.0 * x3 = 1.0
1	LinearInequalityConstraint	1.0 * x1 + 2.0 * x3 <= 0.8
```


## Set up a strategy

To solve the optimization problem, we further need a solving strategy. BoFire supports strategies without a prediction model such as a random strategy and predictive strategies which are based on a prediction model.

All strategies contain an ask method returning a defined number of candidate experiments. 

### Random strategy

```python
from bofire.data_models.strategies.api import RandomStrategy

import bofire.strategies.mapper as strategy_mapper

strategy_data_model = RandomStrategy(domain=domain)

random_strategy = strategy_mapper.map(strategy_data_model)
random_candidates = random_strategy.ask(2)

random_candidates
>>> 	x1	x2	x3	x4	x6	x5
0	0.169125	0.682277	0.148597	1.0	c2	A
1	0.033154	0.873228	0.093617	5.0	c2	A
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
        "valid_y1": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    }
)

X
>>> x1	x2	x3	x4	x5	x6	y1	valid_y1
0	0.0	0.7	0.3	1.0	C	c1	2.00	1
1	0.5	0.5	0.0	2.0	B	c2	4.25	1
2	0.2	0.6	0.2	5.0	A	c1	6.16	1
3	0.1	0.7	0.2	7.5	A	c2	9.59	1
4	0.0	0.7	0.3	2.0	B	c3	5.00	1
5	0.5	0.5	0.0	5.0	B	c3	8.25	1
6	0.2	0.6	0.2	7.5	A	c1	8.66	1
7	0.1	0.7	0.2	1.0	B	c2	3.09	1
8	0.0	0.7	0.3	5.0	A	c3	8.00	1
9	0.5	0.5	0.0	7.5	A	c1	8.75	1
10	0.2	0.6	0.2	1.0	B	c2	3.16	1
11	0.1	0.7	0.2	2.0	A	c3	5.09	1
```

Note that we defined one entry of x5 to be the non-allowed category. If we would not pass historical data containing the non-allowed category, BoFire would suggest to remove the unused category from the domain:

```python
from bofire.data_models.strategies.api import SoboStrategy
from bofire.data_models.acquisition_functions.api import qNEI

sobo_strategy_data_model = SoboStrategy(domain=domain_single_objective, acquisition_function=qNEI())

sobo_strategy = strategy_mapper.map(sobo_strategy_data_model)

sobo_strategy.tell(X)
```

```python
sobo_strategy.ask(candidate_count=1, add_pending=True)
>>> x1	x2	x3	x4	x6	x5	y1_pred	y1_sd	y1_des
0	1.0	0.0	0.0	7.5	c3	A	9.813998	0.703783	9.813998
```
The argument replace = True  determines, if former experiments stored to the domain should be overwritten or the new experiments should be append.