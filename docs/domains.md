# Domains 

The BoFire domain contains all information about the optimization problem. In general, the domain is defined by **inputs**, **outputs**, **objectives** and **constraints**. Numerous types of those elements are implemented in the BoFire framework. A brief description of the usage of domains and the individual elements is given in the following.

A basic example of a domain consists of two continuous inputs $x_1$, $x_2$ and a continuous output $y$. If no objective is given, a maximization objective (TODO: link to later section...) for the output is assumed. If no constraints are defined, an empty set of constraints is assumed.

A continuous input can be defined using the `ContinuousInput` class, which requires the variable name and its bounds. Please note that unbounded and partially bounded input variables are currently not supported. Here, we assume $x_1, x_2 \in (0,1)$
```python
from bofire.data_models.features.api import ContinousInput

inputs = [
    ContinuousInput(key="x1", bounds=(0,1)),
    ContinuousInput(key="x2", bounds=(0,1))
    ]
```
Analogously, the continuous output is defined using the `ContinuousOutput` class.
```python
from bofire.data_models.features.api import ContinousOutput

outputs = [ContinuousOutput(key="y")]
```
In this case, the domain definition is as follows
```python
from domain.data_models.domain.api import Domain

domain = Domain(
    inputs=inputs,
    outputs=outputs
)
```
There are applications of domains, where no optimization of an objective is involved (e.g. randomly sampling from the design space). In such cases, we do not require to provide an outp
Let's now assume we have an additional linear equality constraint and we want to use a custom objective, e.g. a minimization objective for $y$. The linear constraint can be defined using the `LinearEqualityConstraint` class
```python
from bofire.data_models.constraints.api import LinearEqualityConstraint

constraints = [LinearEqualityConstraint(features=["x1", "x2"], coefficients=[1,1], rhs=1)]
```
The code above corresponds to the linear equation

$$
x_1 + x_2 = 1.
$$

In BoFire, the minimization objective can be selected in the output definition using the `MinimizeObjective` class
```python
from bofire.data_models.objectives.api import MinimizeObjective

outputs = [ContinuousOutput(key="y", objective=MinimizeObjective())]
```
The modified domain in this case becomes  
```python
domain = Domain(
    inputs=inputs,
    outputs=outputs,
    constraints=constraints
)

```