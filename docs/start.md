# Getting started

In BoFire, an optimization problem is defined by defining a domain containing input and output features as well as constraints (optional). 

## Features
Input features can be continuous, categorical, or categorical with descriptors:

```python
from bofire.domain.features import ContinuousInput, DiscreteInput, CategoricalInput, CategoricalDescriptorInput

x1 = ContinuousInput(key="x1", lower_bound=0, upper_bound=1)
x2 = ContinuousInput(key="x2", lower_bound=0, upper_bound=1)
x3 = ContinuousInput(key="x3", lower_bound=0, upper_bound=1)
x4 = DiscreteInput(key="x4", values=[1, 2, 5, 7.5])
x5 = CategoricalInput(key="x5", categories=["A", "B", "C"], allowed=[True,True,False])
x6 = CategoricalDescriptorInput(key="x6", categories=["c1", "c2", "c3"], descriptors=["d1", "d2"], values = [[1,2],[2,5],[1,7]])
```

As output features, currently only continuous output features are supported. Each ouput feature has to have an objective, which can be a minimze or maximize objective. Each objective needs to have a weight w between 0 and 1, which would be considered in 
```python
from bofire.domain.features import ContinuousOutput
from bofire.domain.objectives import MaximizeObjective

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
```
input_features.get_keys()
>>> ["x1", "x2", "x3", "x4", "x5"]

output_features.get_keys()
>>> ["y1", "y2"]
```

The input feature container further provides methods to return a feature container with only all fixed or all free features.
```python
free_inputs = input_features.get_free()
fixed_inputs = input_features.get_fixed()
```s

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
s
**Linear constraints** (`LinearEquality` and `LinearInequality`) are expressions of the form $\sum_i a_i x_i = b$ or $\leq b$ for equality and inequality constraints respectively.
They take a list of names of the input features they are operating on, a list of left-hand-side coefficients $a_i$ and a right-hand-side constant $b$.

```python
from bofire.domain.constraints import LinearEqualityConstraint, LinearInequalityConstraint

# A mixture: x1 + x2 + x3 = 1
constr1 = LinearEqualityConstraint(features=["x1", "x2", "x3"], coefficients=[1,1,1], rhs=1)

# x1 + 2 * x3 < 0.8
constr2 = LinearInequalityConstraint(features=["x1", "x3"], coefficients=[1, 2], rhs=0.8)
```
Because of the product $a_i x_i$, linear constraints cannot operate on categorical parameters.

**Nonlinear constraints** (`NonlinearEquality` and `NonlinearInequality`) take any expression that can be evaluated by [pandas.eval](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.eval.html), including mathematical operators such as `sin`, `exp`, `log10` or exponentiation.
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

Finally, there is a **combinatorical constraint** (`NChooseK`) to express that we only want to have $k$ out of the $n$ parameters to take positive values.
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
Finally, the domain can be instanciated:
```python
from bofire.domain.domain import Domain

domain = Domain(
    input_features=input_features, 
    output_features=output_features, 
    constraints=constraints
    )
```
A summary of the defined features and constraints can be obtained by the methods get_feature_reps_df() and get_constraint_reps_df():
```python
domain.get_feature_reps_df()
domain.get_constraint_reps_df()
```


## Set up a strategy
To solve the optimization problem, we further need a solving strategy. BoFire supports strategies without a prediction model such as a random strategy and predictive strategies.

```python
from bofire.strategies.random import RandomStrategy

random_strategy = RandomStrategy(domain=domain)
random_candidates = random_strategy.ask(2)

random_candidates
>>> 	x1	x2	x3	x4	x6	x5
0	0.646864	0.317559	0.035577	1.0	c1	B
1	0.341934	0.525235	0.132830	2.0	c3	A
```

Since a predictive strategy includes a prediction model, we need to generate some historical data:

```python
import random 

for feat in domain.output_features.get_keys():
    X[feat] = [random.uniform(0, 10) for _ in range(10)]
    X["valid_" + feat] = 1

X.at[0,'x5'] = 'C'

X
>>> 
x1	x2	x3	x4	x6	x5	y1	valid_y1	y2	valid_y2	y3	valid_y3
0	0.076580	0.114333	0.908693	2.0	c3	C	1.778685	1	7.431654	1	6.480092	1
1	0.101352	0.816144	0.998311	1.0	c3	A	4.455056	1	4.308399	1	1.115206	1
2	0.786277	0.723443	0.958959	1.0	c1	A	1.409387	1	6.301221	1	1.187027	1
3	0.230341	0.717695	0.080950	5.0	c2	A	1.828613	1	4.099128	1	5.062944	1
4	0.696330	0.404462	0.111464	7.5	c2	B	9.743244	1	0.452868	1	3.402737	1
5	0.434963	0.593474	0.664282	2.0	c3	A	6.041753	1	6.515593	1	9.370194	1
6	0.770037	0.369990	0.879929	1.0	c2	A	0.701850	1	5.059462	1	0.709087	1
7	0.757170	0.494272	0.868837	7.5	c3	A	8.477390	1	6.085909	1	5.291752	1
8	0.487407	0.482285	0.521498	5.0	c3	A	5.454225	1	2.672694	1	5.616514	1
9	0.539006	0.689065	0.333560	5.0	c1	A	6.503267	1	7.491846	1	4.210929	1
```

Note that we defined one entry of x5 to be the non-allowed category. If we would not pass historical data containing the non-allowed category, BoFire would suggest to remove the unused category from the domain:

```python
sobo_strategy.tell(X.loc[1:], replace=True)
```

The argument replace = True  