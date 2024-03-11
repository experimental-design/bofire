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

## Inputs & Outputs
BoFire allows for the following different user-facing input classes (TODO: check if this is true):

- `ContinuousInput`
- `DiscreteInput` 
- `CategoricalInput`
- `MolecularInput`
- `ContinuousDescriptorInput` TODO: Ask Johannes if it is just an alternative name to ContinuousInput
- `CategoricalDescriptorInput` TODO: dito
- `CategoricalMolecularInput`
- `TaskInput`

Each of these input classes can be used for defining domains, however some strategies only support a subset of the available input types. You can call the `is_feature_implemented()` function of a given strategy and input class to check whether the input is supported by the strategy (see also (TODO: link to strategy docu)).

### Continuous inputs 

Continuous inputs are used to define real-values input variables with finite upper and lower bounds (see example above).

### Discrete inputs

For discrete inputs, only a finite set of values is allowed. The `DiscreteInput` class requires a list of permissible values as input. For example, the following code defines a new discrete input variable $x_3$ with values $0, 0.1, 0.2$.

```python
from bofire.data_models.features.api import DiscreteInput

DiscreteInput(key="x3", values=[0, 0.1, 0.2])
```

### Categorical inputs
This class of inputs is similar to the discrete inputs, but takes a list of strings as input. The following code defines a new categorical input variable $x_4$ with values "A", "B", "C".

```python   
from bofire.data_models.features.api import CategoricalInput

CategoricalInput(key="x4", values=["A", "B", "C"])
```

### Molecular inputs
Molecular inputs are used to define molecular structures. This requires the optional cheminfo dependencies installed. The `MolecularInput` class transforms a SMILES input into an [rdkit mol object](https://www.rdkit.org/docs/GettingStartedInPython.html). For example, the following code defines a new molecular input variable $x_5$ with a SMILES string.

```python
from bofire.data_models.features.api import MolecularInput

MolecularInput(key="x5", smiles="CCO")
```


TODO: Inputs() class --> Summarize multiple inputs


## Constraints
## Objectives

In BoFire diffierent types of objectives are implemented. Each output variable can have its own objective. The following objectives are available:

- `MaximizeObjective`: This is the default value. The objective is to maximize the output variable(s).
- `MinimizeObjective`: The objective is to minimize the output variable(s). Note that minimization objectives can be transformed into maximization objectives and vice versa just by multiplying the corresponding output by -1.
- `MaximizeSigmoidObjective`: The objective is to maximize the output variable(s) using a sigmoid transformation. This is useful for objectives that are not linear in the output variable(s).
- `MinimizeSigmoidObjective`: The objective is to minimize the output variable(s) using a sigmoid transformation.
- `TargetObjective`: The objective is to reach a target value for the output variable(s).
- `CloseToTargetObjective`: **TODO**: Ask Johannes what is the difference to TargetObjective and how it is used (multiobjective optimization?)
- `ConstrainedCategoricalObjective`: Categorical objective where for each output variable a probability vector for the categories is given. **TODO**: Ask Johannes if this is correct