# Domains

The BoFire domain contains all information about the optimization problem. In general, the domain is defined by **inputs**, **outputs**, **objectives** and **constraints**. Different types for each of these elements are implemented in the BoFire framework. A brief description of the usage of domains and the individual elements is given in the following.

A basic example of a domain consists of two continuous inputs $x_1$, $x_2$ and a continuous output $y$. If no objective is given, a maximization objective for the output is assumed. If no constraints are defined, an empty set of constraints is assumed.

A continuous input can be defined using the `ContinuousInput` class, which requires the variable name and its bounds. Please note that unbounded and partially bounded input variables are currently not supported. Here, we assume $x_1, x_2 \in [0,1]$.
```python
from bofire.data_models.features.api import ContinuousInput
from bofire.data_models.domain.api import Inputs

inputs = Inputs(features=[
    ContinuousInput(key="x1", bounds=[0,1]),
    ContinuousInput(key="x2", bounds=[0,1])
]
)
```
Analogously, the continuous output is defined using the `ContinuousOutput` class.
```python
from bofire.data_models.features.api import ContinuousOutput
from bofire.data_models.domain.api import Outputs

outputs = Outputs(features=[
    ContinuousOutput(key="y")
    ]
)
```
In this case, the domain definition is as follows
```python
from bofire.data_models.domain.api import Domain

domain = Domain(
    inputs=inputs,
    outputs=outputs
)
```
Domains can also be created more concisely using the `from_lists` method of the `Domain` class. This is especially useful if you want to define a domain without having to create separate lists for inputs and outputs. The following code achieves the same result as above:
```python
domain = Domain.from_lists(
    inputs=[
        ContinuousInput(key="x1", bounds=[0,1]),
        ContinuousInput(key="x2", bounds=[0,1])
    ],
    outputs=[
        ContinuousOutput(key="y")
    ]
)
```
There are applications of domains, where no optimization of an objective is involved (e.g. randomly sampling from the design space). In such cases, we do not require to provide an output definition.

Let us now assume, we have an additional linear equality constraint and we want to use a custom objective, e.g. a minimization objective for $y$. The linear constraint can be defined using the `LinearEqualityConstraint` class
```python
from bofire.data_models.constraints.api import LinearEqualityConstraint

constraints = [LinearEqualityConstraint(features=["x1", "x2"], coefficients=[1,1], rhs=1)]
```
The code above corresponds to the linear equation

$$
x_1 + x_2 = 1.
$$

In BoFire, the minimization objective can be selected by setting the objective-attribute of an output to an instance of `MinimizeObjective`
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

## Inputs
BoFire allows for the following different user-facing input classes:

- `ContinuousInput`
- `DiscreteInput`
- `CategoricalInput`
- `MolecularInput`
- `CategoricalDescriptorInput`
- `CategoricalMolecularInput`
- `TaskInput`
<!-- - `ContinuousDescriptorInput` not used atm -->

Each of these input classes can be used for defining domains, however some strategies only support a subset of the available input types. You can call the `is_feature_implemented()` function of a given strategy and input class to check whether the input is supported by the strategy. For example, the following code checks whether the `ContinuousInput` is supported by the `RandomSearch` strategy.
<!-- link to strategy docu shall be added once it is there -->

```python
from bofire.data_models.strategies.api import RandomStrategy

RandomStrategy.is_feature_implemented(ContinuousInput)
```
The result will be `True` if the input type is supported by the strategy, otherwise it will return `False`. This is useful to check whether a certain input type can be used with a specific strategy before defining the domain.

### Input types

#### Continuous inputs

Continuous inputs are used to define real-valued input variables with finite upper and lower bounds (see example above).

#### Discrete inputs

For discrete inputs, only a finite set of values is allowed. The `DiscreteInput` class requires a list of permissible values as input. For example, the following code defines a new discrete input variable $x_3$ with values $0, 0.1, 0.2$.

```python
from bofire.data_models.features.api import DiscreteInput

DiscreteInput(key="x3", values=[0, 0.1, 0.2])
```

#### Categorical inputs
This class of inputs is similar to the discrete inputs, but takes a list of strings as input. The following code defines a new categorical input variable $x_4$ with categories "A", "B", "C".

```python
from bofire.data_models.features.api import CategoricalInput

CategoricalInput(key="x4", categories=["A", "B", "C"])
```

#### Molecular inputs
Molecular inputs are used to define molecular structures. This requires the optional cheminfo dependencies installed. The `MolecularInput` class operates on SMILES strings. For example, the following code defines a new molecular input variable $x_5$.

```python
from bofire.data_models.features.api import MolecularInput

MolecularInput(key="x5")
```

<!-- #### Continuous descriptor inputs
 -->
#### Categorical descriptor inputs
Via the `CategoricalDescriptorInput` one can provide continuous encodings for the different categories via so called descriptors. Imagine, for example, having a categorial input with different categories, where each category corresponds to a specific material. Such descriptors could be e.g., density and hardness. Every material/category would get assigned a number for density and hardness in the hope that these two properties describe the material properly. In the context of fitting a GP, one can then use just these two dimensional vector for describing the material instead of a ten dimensional one-hot encoding, which results in a dimensionality reduction. Of course, this makes sense under the assumption that the descriptors actually correlate with the desired quantities.

The `CategoricalDescriptorInput` class requires a list of permissible values as input. For example, the following code defines a new categorical descriptor input variable $x_6$ with values "A", "B", "C".

```python
from bofire.data_models.features.api import CategoricalDescriptorInput

CategoricalDescriptorInput(
    key="x6",
    categories=["material_A", "material_B", "material_C"],
    descriptors=["density", "hardness"],
    values=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
)

```

#### Categorical molecular inputs
Categorical molecular inputs inherit from both molecular and categorical inputs. In addition to the 'MolecularInput' the user can provide a list of allowed SMILES strings via the `categories` attribute. The following code defines a new categorical molecular input variable $x_7$ where only the values "C1CCCCC1", "O1CCOCC1" are allowed.

```python
from bofire.data_models.features.api import CategoricalMolecularInput

CategoricalMolecularInput(key="x7", categories=["C1CCCCC1", "O1CCOCC1"])
```

<!--#### Task inputs
 -->

### Inputs class
The `Inputs` class is used to collect multiple input variables. It is used to define the inputs of a domain. The following code defines a new input class with the above described input variables $x_1, x_2, x_3, x_4, x_5, x_6, x_7$.

```python
from bofire.data_models.api import Inputs

inputs = Inputs(
    features=[ContinuousInput(key="x1", bounds=[0,1]),
    ContinuousInput(key="x2", bounds=[0,1]),
    DiscreteInput(key="x3", values=[0, 0.1, 0.2]),
    CategoricalInput(key="x4", categories=["A", "B", "C"]),
    MolecularInput(key="x5"),
    CategoricalDescriptorInput(key="x6", categories=["material_A", "material_B", "material_C"], descriptors=["density", "hardness"], values=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
    CategoricalMolecularInput(key="x7", categories=["C1CCCCC1", "O1CCOCC1"])])
```

## Outputs
At the moment only continuous and categorical outputs are supported. Those are similar to the continuous and categorical inputs, but they additionaly contain the `objective` attribute. The `objective` attribute is used to define the optimization objective for the output variable. Similar to the inputs the outputs can be also collected in an `Outputs` object. An example with a continuous and a categorical output is given below.

```python
from bofire.data_models.api import Outputs
from bofire.data_models.objectives.api import ConstrainedCategoricalObjective
from bofire.data_models.features.api import ContinuousOutput, CategoricalOutput

outputs = Outputs(
    features=[
        ContinuousOutput(key="y1", objective=MinimizeObjective()),
        CategoricalOutput(
            key="y2",
            categories=["a", "b", "c"],  # Add the required categories
            objective=ConstrainedCategoricalObjective(
                categories=["a", "b", "c"],
                desirability=[True, False, True]
            )
        )
    ]
)
```
### Objectives
Different classes for the objectives are implemented in BoFire. These are used to set the `objective` attribute of an output object. Note that each output variable can have its own objective. The following objectives are available:

- `MaximizeObjective`: This is the class of the default value. The objective is to maximize the output variable(s).
- `MinimizeObjective`: The objective is to minimize the output variable(s). Note that minimization objectives can be transformed into maximization objectives and vice versa just by multiplying the corresponding output by -1.
- `MaximizeSigmoidObjective`: The objective is to maximize the output variable(s) using a sigmoid transformation. This is useful to implement bounds on the output variable(s).
- `MinimizeSigmoidObjective`: Similar to `MaximizeSigmoidObjective`, but the objective is to minimize the output variable(s).
- `TargetObjective`: The objective is to reach a target value for the output variable(s). `TargetObjective` is of type `ConstrainedObjective` as `MaximizeSigmoidObjective` or `MinimizeSigmoidObjective`, i.e., it becomes one if the value is in the target region and falls asymptorically to zero outside that region. That means that once the target objective is met, e.g., objective value above target value in a `MaximizeSigmoidObjective` case, the optimization does not care about how close the objective value is compared to the target value.
- `CloseToTargetObjective`: This objective measures the difference to a target value. Such an objective is often meaningful to minimize in a multiobjective optimization, and thus be included in the pareto front. Note that also the objectives of type `ConstrainedObjective` can be used in multiobjective optimization, but for that at least two targets of type `MaximizeSigmoidObjective`,`MinimizeSigmoidObjective` or `TargetObjective` are needed. The main difference with respect to the `TargetObjective` is that here the goal is to stay as close as possible to the target value.
- `ConstrainedCategoricalObjective`: Categorical objective where for each output variable a probability vector for the categories is given.


If many outputs are defined, then different weights can be attributed as optional argument `w`. More detailed descriptions for the individual objecrives can be found in the [API documentation](https://experimental-design.github.io/bofire/ref-objectives/).

## Constraints
The optimization problem in BoFire can be solved subject to a variety of different constraint types in the input space. These constraints are currently categorized as linear, non-linear, n choose k and interpoint constraints.

### Linear constraints
Linear constraints are used to define linear equalities and inequalities.

Equalities can be defined using the `LinearEqualityConstraint` class. The `features` attribute is used to define the input variables, the `coefficients` attribute is used to define the coefficients of the linear equation and the `rhs` attribute is used to define the right-hand side of the equation.

Inequalities can be defined using the `LinearInequalityConstraint` class. The attributes are defined in the same way as for the `LinearEqualityConstraint` class.

The following code defines a new linear equality constraint $x_1 + x_2 = 1$ and an inequality constraint $x_1 - x_3 \leq 1$.

```python

from bofire.data_models.constraints.api import LinearEqualityConstraint, LinearInequalityConstraint

constraints = [LinearEqualityConstraint(features=["x1", "x2"], coefficients=[1,1], rhs=1),
               LinearInequalityConstraint(features=["x1", "x3"], coefficients=[1,-1], rhs=1)]
```

Note that the variables in the `features` attribute used in the constraints must be previously defined as inputs in the domain, in order to have a well defined optimization problem.

### Non-linear constraints
Non-linear constraints can be used to define equalities and inequalities of the form:

$$
c(\mathbf{x}) = 0 \quad \text{and} \quad c(\mathbf{x}) \leq 0,
$$

where $c(\mathbf{x})$ can be an any differentiable function of the inputs **x** represented by a string attribute `expression` that can be evaluated via the `eval()` method of pandas dataframe. If `sympy` is installed, the derivate expressions are automatically calculated from the `expression` attribute. Otherwise, the user can provide additional expressions for the derivatives using the `jacobian_expressions` attribute. The `features` attrribute should contain the names of the input variables used in the `expression` attribute.

The following code defines a new non-linear inequality constraint $x1^2 + x2^2 - x3 \leq 0$.

```python
from bofire.data_models.constraints.api import NonlinearInequalityConstraint

NonlinearInequalityConstraint(expression="x1**2 + x2**2 - x3", features=["x1","x2","x3"])
```

### N choose k constraints
Given a list of N features and attributes `min_count`, `max_count`, the `NChooseKConstraint` class is used to define constraints that require a minimum and maximum number of features to be nonzero. Note that one can put min_count==max_count. The following code defines a new `NChooseKConstraint` that requires at least 2 and at most 3 features to be selected from the list ["x1", "x2", "x3", "x4"].

```python
from bofire.data_models.constraints.api import NChooseKConstraint

NChooseKConstraint(features=["x1", "x2", "x3", "x4"], min_count=2, max_count=3, none_also_valid=False)
```

### Interpoint constraints
The `InterpointEqualityConstraint` forces that values of a certain feature of a set/batch of candidates should have the same value. The set is defined using the `multiplicity` attribute. The following code defines a new `InterpointEqualityConstraint` that forces the values of the feature "x1" to be the same for every 3 subsequent candidates.

```python
from bofire.data_models.constraints.api import InterpointEqualityConstraint

InterpointEqualityConstraint(features=["x1"], multiplicity=3)
```
