#%%
import pandas as pd
import numpy as np
from itertools import product
#%%
from bofire.data_models.domain import api as domain_api
from bofire.data_models.features import api as features_api
from bofire.data_models.constraints import api as constraints_api
from bofire.data_models.objectives import api as objectives_api
#%% md
# Domain definition
#%%
domain = domain_api.Domain(
    inputs=domain_api.Inputs(features=[
        features_api.ContinuousInput(key="temperature", bounds=(30.0, 100.0)),
        features_api.CategoricalInput(key="solvent", categories=["water", "ethanol"]),
    ]),
    outputs=domain_api.Outputs(features=[
        features_api.ContinuousOutput(key="yield", objective=objectives_api.MaximizeObjective()),
    ]),
    constraints=domain_api.Constraints(constraints=[
        constraints_api.CategoricalExcludeConstraint(
            features=["solvent", "temperature"],
            conditions=[
                constraints_api.SelectionCondition(selection=["ethanol"]),
                constraints_api.ThresholdCondition(threshold=70.0, operator=">="),
            ]
        ),
    ])
)
#%%
print(domain)
#%% md
## Generate candidates and check constraint fulfillment
#%%
candidates = pd.DataFrame(
    list(
        product(
            [30.0, 50.0, 70.0, 90.0],
            ["water", "ethanol"],
        )
    ),
    columns=["temperature", "solvent"],
)
#%%
candidates["valid"] = domain.constraints.is_fulfilled(candidates, tol=1e-5)
print(candidates)

#%% md
# BO workflow with constraints
#%%
data = candidates[candidates["valid"]].drop(columns=["valid"])
data["yield"] = np.random.rand(len(data))
#%%
from bofire.data_models.strategies import api as strategies_api
from bofire.data_models.acquisition_functions import api as af_api
from bofire.strategies import api as strategies_api_impl

#%%
optimizer = strategies_api.GeneticAlgorithmOptimizer(verbose=True)
#%%
strategy = strategies_api_impl.SoboStrategy.make(
    domain=domain,
    acquisition_function=af_api.qUCB(),
    acquisition_optimizer=optimizer,
)

#%%

strategy.tell(data)
#%%
candidates = strategy.ask(30)
print(candidates)