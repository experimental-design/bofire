# Data Models


## General

`bofire` separates Data Models from the actual functionality. Data models exist for all relevant objects like the `Domain` and its contents, `Strategies` and `Surrogates`. 
Each optimization endeavour in `bofire` starts by creating the relevant data models, followed by mapping them to the respective objects that also hold the functionality.

All data models are based on the [`pydantic`](https://docs.pydantic.dev/latest/) library. `pydantic` allows for easy (de-)serialization 
of data models, and automatically validates & converts the data types.

### Use in Web Frameworks

Additionally, it enables a fast integration into web frameworks, such as e.g. 
FastAPI by simply importing the data models from the `bofire.data_models`
submodule. (see example below)
We currently do not provide working web frontend and backend examples, but have this planned for the future. 


### Lightweight installation

The data models can be installed separately from the main package. by installing via `pip install bofire`, instead of  `pip install bofire[optimization]`, you can install the data models only. 


## Creating Data Model Objects

Data models can be found in the `bofire.data_models` submodule, and each submodule features a collection of all data-models of this submodule in `*.api`. Below you find some examples uses of it:

```python
# domain specific data models
import bofire.data_models.domain.api as domain_api
import bofire.data_models.features.api as features_api
import bofire.data_models.constraints.api as constraints_api
import bofire.data_models.objectives.api as objectives_api

# individual inputs
in_x1 = features_api.ContinuousInput(key="x1", bounds=(-4, 4))
in_x2 = features_api.CategoricalInput(key="x2", categories=["a", "b"])

# Inputs object
input_features = domain_api.Inputs(features=[in_x1, in_x2])

# individual outputs
objective_y1 = objectives_api.MaximizeObjective(bounds=(0, 100))
out_y1 = features_api.ContinuousOutput(key="y1",objective=objective_y1)

# Outputs object
output_features = domain_api.Outputs(features=[out_y1],)

# Domain
domain = domain_api.Domain(inputs=input_features, outputs=output_features)
```
This domain object can now be saved, passed to a strategy, or used in a surrogate. 

* <details>
  <summary><code>domain.dict()</code> to yield a dictionary</summary>
    <pre><code>
    # printout of domain.dict()
    { 'constraints': {'constraints': [], 'type': 'Constraints'},
        'inputs': { 'features': [ { 'bounds': (-4.0, 4.0),
                                    'key': 'x1',
                                    'local_relative_bounds': (inf, inf),
                                    'stepsize': None,
                                    'type': 'ContinuousInput',
                                    'unit': None},
                                { 'allowed': [True, True],
                                    'categories': ['a', 'b'],
                                    'key': 'x2',
                                    'type': 'CategoricalInput'}],
                    'type': 'Inputs'},
        'outputs': { 'features': [ { 'key': 'y1',
                                    'objective': { 'bounds': (0.0, 100.0),
                                                    'type': 'MaximizeObjective',
                                                    'w': 1},
                                    'type': 'ContinuousOutput',
                                    'unit': None}],
                    'type': 'Outputs'},
        'type': 'Domain'}
    </code></pre>
</details>

* <details>
  <summary><code>domain.json()</code> to yield a json string</summary>
    <pre><code>
    # printout of domain.json(); as domain.model_dump_json(indent=2)
    {
    "type": "Domain",
    "inputs": {
        "type": "Inputs",
        "features": [
        {
            "type": "ContinuousInput",
            "key": "x1",
            "unit": null,
            "bounds": [
            -4.0,
            4.0
            ],
            "local_relative_bounds": [
            Infinity,
            Infinity
            ],
            "stepsize": null
        },
        {
            "type": "CategoricalInput",
            "key": "x2",
            "categories": [
            "a",
            "b"
            ],
            "allowed": [
            true,
            true
            ]
        }
        ]
    },
    "outputs": {
        "type": "Outputs",
        "features": [
        {
            "type": "ContinuousOutput",
            "key": "y1",
            "unit": null,
            "objective": {
            "type": "MaximizeObjective",
            "w": 1.0,
            "bounds": [
                0.0,
                100.0
            ]
            }
        }
        ]
    },
    "constraints": {
        "type": "Constraints",
        "constraints": []
    }
    }
    </code></pre>
</details>

Note that the data contains many more fields compared to what it has been initialized with. This is due to the defaults that are set with every data model.

A serialized model can be recovered in various ways (some of which are deprecated as of pydantic 2):
```python
# save domain
import json
domain_as_json_string = domain.json()

# option0:
from pydantic import TypeAdapter
recovered_domain = TypeAdapter(domain_api.Domain).validate_python(json.loads(domain_as_json_string))

# option 1:  parse_obj_as
from pydantic import parse_obj_as
recovered_domain = parse_obj_as(domain_api.Domain, json.loads(domain_as_json_string))

# option 2: from dictionary via kwargs
domain_dict = json.loads(domain_as_json_string)
recovered_domain = domain_api.Domain(**domain_dict)

# option 3:
domain.parse_obj(json.loads(domain_as_json_string))
domain.parse_raw(domain_as_json_string)
```

In all cases,
you can check that the domains are identical via

```python 
>>> recovered_domain == domain
True
```



# SOBO 
```python 
import bofire.data_models.strategy.api as strategy_api
import bofire.data_models.surrogates.api as surrogates_api

strategy = strategy_api.SoboStrategy(domain=domain)
```
Again, strategy is augmented with lots of default values and hyperparamters. 

<details> strategy.dict() 
    <pre><code>
{
  "type": "SoboStrategy",
  "domain": {
    "type": "Domain",
    "inputs": {
      "type": "Inputs",
      "features": [
        {
          "type": "ContinuousInput",
          "key": "x1",
          "unit": null,
          "bounds": [
            -4.0,
            4.0
          ],
          "local_relative_bounds": [
            Infinity,
            Infinity
          ],
          "stepsize": null
        },
        {
          "type": "CategoricalInput",
          "key": "x2",
          "categories": [
            "a",
            "b"
          ],
          "allowed": [
            true,
            true
          ]
        }
      ]
    },
    "outputs": {
      "type": "Outputs",
      "features": [
        {
          "type": "ContinuousOutput",
          "key": "y1",
          "unit": null,
          "objective": {
            "type": "MaximizeObjective",
            "w": 1.0,
            "bounds": [
              0.0,
              100.0
            ]
          }
        }
      ]
    },
    "constraints": {
      "type": "Constraints",
      "constraints": []
    }
  },
  "seed": null,
  "num_sobol_samples": 512,
  "num_restarts": 8,
  "num_raw_samples": 1024,
  "descriptor_method": "EXHAUSTIVE",
  "categorical_method": "EXHAUSTIVE",
  "discrete_method": "EXHAUSTIVE",
  "surrogate_specs": {
    "surrogates": [
      {
        "hyperconfig": {
          "type": "MixedSingleTaskGPHyperconfig",
          "hyperstrategy": "FactorialStrategy",
          "inputs": {
            "type": "Inputs",
            "features": [
              {
                "type": "CategoricalInput",
                "key": "continuous_kernel",
                "categories": [
                  "rbf",
                  "matern_1.5",
                  "matern_2.5"
                ],
                "allowed": [
                  true,
                  true,
                  true
                ]
              },
              {
                "type": "CategoricalInput",
                "key": "prior",
                "categories": [
                  "mbo",
                  "botorch"
                ],
                "allowed": [
                  true,
                  true
                ]
              },
              {
                "type": "CategoricalInput",
                "key": "ard",
                "categories": [
                  "True",
                  "False"
                ],
                "allowed": [
                  true,
                  true
                ]
              }
            ]
          },
          "n_iterations": null,
          "target_metric": "MAE"
        },
        "aggregations": null,
        "type": "MixedSingleTaskGPSurrogate",
        "inputs": {
          "type": "Inputs",
          "features": [
            {
              "type": "ContinuousInput",
              "key": "x1",
              "unit": null,
              "bounds": [
                -4.0,
                4.0
              ],
              "local_relative_bounds": [
                Infinity,
                Infinity
              ],
              "stepsize": null
            },
            {
              "type": "CategoricalInput",
              "key": "x2",
              "categories": [
                "a",
                "b"
              ],
              "allowed": [
                true,
                true
              ]
            }
          ]
        },
        "outputs": {
          "type": "Outputs",
          "features": [
            {
              "type": "ContinuousOutput",
              "key": "y1",
              "unit": null,
              "objective": {
                "type": "MaximizeObjective",
                "w": 1.0,
                "bounds": [
                  0.0,
                  100.0
                ]
              }
            }
          ]
        },
        "input_preprocessing_specs": {
          "x2": "ONE_HOT"
        },
        "dump": null,
        "scaler": "NORMALIZE",
        "output_scaler": "STANDARDIZE",
        "continuous_kernel": {
          "type": "MaternKernel",
          "ard": true,
          "nu": 2.5,
          "lengthscale_prior": {
            "type": "GammaPrior",
            "concentration": 3.0,
            "rate": 6.0
          }
        },
        "categorical_kernel": {
          "type": "HammondDistanceKernel",
          "ard": true
        },
        "noise_prior": {
          "type": "GammaPrior",
          "concentration": 1.1,
          "rate": 0.05
        }
      }
    ]
  },
  "outlier_detection_specs": null,
  "min_experiments_before_outlier_check": 1,
  "frequency_check": 1,
  "frequency_hyperopt": 0,
  "folds": 5,
  "local_search_config": null,
  "acquisition_function": {
    "type": "qLogNEI",
    "prune_baseline": true
  }
}
    </code></pre>
</details>

## Customization

In order to use non-default surrogate and strategy options, you will always have to create the respective data model from scratch, and built higher level data structures by passing the respective data models as arguments. as in the example below.

```python
from bofire.data_models.kernels.api import MaternKernel, AdditiveKernel
from bofire.data_models.priors.api import MBO_LENGTHCALE_PRIOR, MBO_NOISE_PRIOR

kernel = MaternKernel(nu=2.5, ard=True, lengthscale_prior=MBO_LENGTHCALE_PRIOR())

surrogate = surrogates_api.MixedSingleTaskGPSurrogate(
    inputs=domain.inputs,
    outputs=domain.outputs,
    continuous_kernel=kernel,
    noise_prior=MBO_NOISE_PRIOR()
)

surrogates = surrogates_api.BotorchSurrogates(surrogates=[surrogate])

custom_strategy = strategy_api.SoboStrategy(
    domain=domain,
    surrogate_specs=surrogates
)

>>> strategy == custom_strategy
False
``` 
Both strategies differ only in the choice of the kernel lengthscale and gp noise prior. 

 



## Integration into Web Frameworks

### FastAPI

Since `fastapi` uses `pydantic`, the data models can be used directly in the API definition. 

Save this to a `fastapi_example.py` and run it with `python fastapi_example.py`. Navigate to `http://localhost:8880/docs` via the browser of your choosing to see the API documentation and try out the route. Note that all the data models (Schemas) are listed in the lower part of the page.

```python
# saving this to file (e.g. to fastapi_minimal.py), and running the fastapi server will yield a working api endpoint with openapi documentation, which is very useful for browsing through!

from fastapi import FastAPI
from bofire.data_models.domain.api import Domain
import bofire.data_models.domain.api as domain_api
from starlette.responses import RedirectResponse
from bofire.data_models.strategies.api import SoboStrategy, AnyStrategy, RandomStrategy
from typing import Union, Optional
from fastapi import Body, Response
import numpy as np
import json

app = FastAPI()
# this string was obtained from the domain above via domain.json()
# local_relative_bounds are set to 200,200 for now, as Infinity could not be parsed properly, and NOne also does not work for now. We'll fix this in the near future!
domain_as_json = '{"type":"Domain","inputs":{"type":"Inputs","features":[{"type":"ContinuousInput","key":"x1","unit":null,"bounds":[-4.0,4.0],"local_relative_bounds":[200,200],"stepsize":null},{"type":"CategoricalInput","key":"x2","categories":["a","b"],"allowed":[true,true]}]},"outputs":{"type":"Outputs","features":[{"type":"ContinuousOutput","key":"y1","unit":null,"objective":{"type":"MaximizeObjective","w":1.0,"bounds":[0.0,100.0]}}]},"constraints":{"type":"Constraints","constraints":[]}}'
# model_config = {"json_schema_extra": {"examples": [{'domain':json.loads(domain_as_json)}]}}
model_config = {"json_schema_extra": {"examples": [json.loads(domain_as_json)]}}
Domain.model_config = model_config
from pydantic import TypeAdapter


# create a single route that returns a SoboStrategy or RandomStrategy based on a coin flip
@app.post(
    "/sobo_or_random_strategy/",
    response_model=Optional[Union[SoboStrategy, RandomStrategy]],
    description="This takes a domain as input and outputs either a SoboStrategy or a RandomStrategy, based on a coin flip. It's meant mainly for didactic purposes. Note also the data models in the 'Schemas' section of the API documentation.",
)
def strategy_post(
    domain: Domain = Body(
        embed=True,
    )
):
    domain = TypeAdapter(domain_api.Domain).validate_python(domain)
    if np.random.normal() >= 0:
        strategy = SoboStrategy(domain=domain)
    else:
        strategy = RandomStrategy(domain=domain)

    return strategy.dict()


@app.get("/", include_in_schema=False)
def redirect():
    """Redirect to the API documentation."""
    return RedirectResponse(url="/docs")


if __name__ == "__main__":
    print("Please run this fastapi server via:")
    print("uvicorn fastapi_minimal:app --reload --port 8880 --host localhost")

# run the server with `uvicorn fastapi_minimal:app --reload --port 8880 --host localhost`
# navigate to http://localhost:8880/docs to see the API documentation

```

```bash
uvicorn fastapi_minimal:app --reload --port 8880 --host localhost
```