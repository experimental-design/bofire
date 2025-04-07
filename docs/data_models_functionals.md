# Data Models vs. Functional Components

Data models in BoFire hold static data of an optimization problem. These are input and output features as well as constraints making up the domain. They further include possible optimization objectives, acquisition functions, and kernels.

All data models in ```bofire.data_models```, are specified as pydantic models and inherit from ```bofire.data_models.base.BaseModel```. These data models can be (de)serialized via ```.dict()``` and ```.model_dump_json()``` (provided by pydantic). A json schema of each data model can be obtained using ```.schema()```.

For surrogates and strategies, all functional parts are located in ```bofire.surrogates``` and ```bofire.strategies```. These functionalities include the ```ask``` and ```tell``` as well as ```fit``` and ```predict``` methods. All class attributes (used by these method) are also removed from the data models. Each functional entity is initialized using the corresponding data model. As an example, consider the following data model of a ```RandomStrategy```:

```python
import bofire.data_models.domain.api as dm_domain
import bofire.data_models.features.api as dm_features
import bofire.data_models.strategies.api as dm_strategies

in1 = dm_features.ContinuousInput(key="in1", bounds=[0.0,1.0])
in2 = dm_features.ContinuousInput(key="in2", bounds=[0.0,2.0])
in3 = dm_features.ContinuousInput(key="in3", bounds=[0.0,3.0])

out1 = dm_features.ContinuousOutput(key="out1")

inputs = dm_domain.Inputs(features=[in1, in2, in3])
outputs = dm_domain.Outputs(features=[out1])
constraints = dm_domain.Constraints()

domain = dm_domain.Domain(
    inputs=inputs,
    outputs=outputs,
    constraints=constraints,
)

data_model = dm_strategies.RandomStrategy(domain=domain)
```

Such a data model can be (de)serialized as follows:

```python
from pydantic import TypeAdapter
from bofire.data_models.strategies.api import AnyStrategy

serialized = data_model.model_dump_json()

data_model_ = TypeAdapter(AnyStrategy).validate_json(serialized)

assert data_model_ == data_model
```
The data model of a strategy contains its hyperparameters.
Using this data model of a strategy, we can create an instance of a (functional) strategy:

```python
import bofire.strategies.api as strategies
strategy = strategies.RandomStrategy(data_model=data_model)
```

As each strategy data model should be mapped to a specific (functional) strategy, we provide such a mapping:

```python
strategy = strategies.map(data_model)
```
