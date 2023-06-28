import os
os.chdir(r'C:\Users\S31015\python_projects\bofire_custom_objectives\bofire')
import sys
sys.path.append(os.getcwd())

import bofire.data_models.domain.api as dm_domain
import bofire.data_models.features.api as dm_features
import bofire.data_models.strategies.api as dm_strategies
import bofire.data_models.acquisition_functions.api as dm_acquisition_functions
import bofire.data_models.objectives.api as dm_objectives
import bofire.data_models.constraints.api as dm_constraints

import bofire.strategies.api as strategies_api

from pydantic import parse_obj_as, BaseModel, Field

import pandas as pd
import numpy as np

import torch
import json
import cloudpickle
import base64

# Import data for ML model fitting
df = pd.DataFrame(np.array(
    [[ 10.0,  15.0, 75.0, 25.0,  30.0,  0.1],
     [ 20.0,  20.0, 60.0,  10.0, 25.0,  0.4],
     [ 40.0,  25.0, 35.0, 15.0,  20.0,  0.6]]),
    columns=['in_a', 'in_b', 'in_c', 'out_a', 'out_b', 'out_c'],
)

X_columns = ['in_a', 'in_b', 'in_c']
Y_columns = ['out_a', 'out_b', 'out_c']

experiments = df[X_columns + Y_columns]

for y in Y_columns:
    experiments[f"valid_{y}"] = 1

input_features = dm_domain.Inputs(features=[dm_features.ContinuousInput(key=f"{x}", bounds=(0, 100)) for x in X_columns])
output_features = dm_domain.Outputs(features=[dm_features.ContinuousOutput(key=f"{y}", objective=dm_objectives.MaximizeObjective()) for y in Y_columns])
constraints = dm_domain.Constraints(constraints=[dm_constraints.LinearEqualityConstraint(features=['in_a', 'in_b', 'in_c'], coefficients=[1.0, 1.0, 1.0], rhs=100)])

domain = dm_domain.Domain(
    inputs=input_features,
    outputs=output_features,
    constraints=constraints,
)

def f(samples: torch.Tensor) -> torch.Tensor:
    return (samples[..., 0] + samples[..., 1]) * (samples[..., 0] + samples[..., 2])

f_bytes_dump = cloudpickle.dumps(f)
f_str_dump = base64.b64encode(f_bytes_dump).decode()

strategy_data_model1 = dm_strategies.CustomSoboStrategy(
    domain=domain,
    acquisition_function=dm_acquisition_functions.qNEI(),
    num_raw_samples=8,
    num_restarts=2,
    num_sobol_samples=8,
)

strategy_data_model1_dumps = dm_strategies.CustomSoboStrategy(
    domain=domain,
    acquisition_function=dm_acquisition_functions.qNEI(),
    num_raw_samples=8,
    num_restarts=2,
    num_sobol_samples=8,
    dump=f_str_dump,
)

strategy_jspec1 = strategy_data_model1.json()
strategy_jspec1_dumps = strategy_data_model1_dumps.json()

strategy_data_model2 = parse_obj_as(dm_strategies.AnyStrategy, json.loads(strategy_jspec1))
strategy_data_model2_dumps = parse_obj_as(dm_strategies.AnyStrategy, json.loads(strategy_jspec1_dumps))

assert(strategy_data_model1 == strategy_data_model2)
assert(strategy_data_model1_dumps == strategy_data_model2_dumps)

strategy1 = strategies_api.map(strategy_data_model1)
strategy2 = strategies_api.map(strategy_data_model2)
strategy2_dumps = strategies_api.map(strategy_data_model2_dumps)

# attach the actual f to them
strategy1.loads(f_str_dump)
strategy2.f = f
assert(isinstance(strategy2_dumps.f, type(f)))

strategy1.tell(experiments)
strategy2.tell(experiments)
strategy2_dumps.tell(experiments)

candidates1 = strategy1.ask(candidate_count=2)
candidates2 = strategy2.ask(candidate_count=2)
candidates2_dumps = strategy2_dumps.ask(candidate_count=2)

print()