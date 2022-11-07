import json
from typing import Dict, Optional

import pandas as pd

from bofire.domain import Domain
from bofire.domain.constraints import (
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
    NonlinearEqualityConstraint,
    NonlinearInqualityConstraint,
)
from bofire.domain.desirability_functions import (
    CloseToTargetDesirabilityFunction,
    MaxIdentityDesirabilityFunction,
    MinIdentityDesirabilityFunction,
)
from bofire.domain.features import (
    CategoricalInputFeature,
    ContinuousInputFeature,
    ContinuousOutputFeature,
    DiscreteInputFeature,
    InputFeature,
    OutputFeature,
)


def input2feature(config: Dict):
    if config["type"] == "continuous":
        return ContinuousInputFeature(
            key=config["name"],
            lower_bound=config["domain"][0],
            upper_bound=config["domain"][1],
        )
    if config["type"] == "categorical":
        return CategoricalInputFeature(key=config["name"], categories=config["domain"])
    if config["type"] == "discrete":
        return DiscreteInputFeature(key=config["name"], values=config["domain"])
    else:
        raise ValueError(f"Unknown parameter type {config['type']}.")


def objective2feature(config: Dict):
    if config["type"] == "minimize":
        d = MinIdentityDesirabilityFunction(w=1.0)
    elif config["type"] == "maximize":
        d = MaxIdentityDesirabilityFunction(w=1.0)
    elif config["type"] == "close-to-target":
        d = CloseToTargetDesirabilityFunction(
            w=1,
            target_value=config["target"],
            exponent=config.get("tolerance", 1.0),
            tolerance=config.get("tolerance", 0.0),
        )
    else:
        raise ValueError(f"Unknown objective type {config['type']}.")
    return ContinuousOutputFeature(key=config["name"], desirability_function=d)


def constraint2constraint(config: Dict, input_feature_keys: Optional[list] = None):
    if config["type"] == "linear-equality":
        return LinearEqualityConstraint(
            features=config["names"], coefficients=config["lhs"], rhs=config["rhs"]
        )
    if config["type"] == "linear-inequality":
        return LinearInequalityConstraint(
            features=config["names"], coefficients=config["lhs"], rhs=config["rhs"]
        )
    if config["type"] == "n-choose-k":
        return NChooseKConstraint(
            features=config["names"],
            max_count=config["max_active"],
            min_count=0,
            none_also_valid=False,
        )
    if config["type"] == "non-linear-equality":
        return NonlinearEqualityConstraint(
            features=input_feature_keys, expression=config["expression"]
        )
    if config["type"] == "non-linear-inequality":
        return NonlinearInqualityConstraint(
            features=input_feature_keys, expression=config["expression"]
        )
    raise ValueError(f"Unknown constraint type {config['type']}.")


def problem2domain(config: Dict):
    input_features = []
    output_features = []
    for pconfig in config["inputs"]:
        input_features.append(input2feature(pconfig))
    for pconfig, oconfig in zip(config["outputs"], config["objectives"]):
        output_features.append(objective2feature(oconfig))
    domain = Domain(input_features=input_features, output_features=output_features)
    if "constraints" in config:
        for cconfig in config["constraints"]:
            domain.add_constraint(
                constraint2constraint(cconfig, domain.get_feature_keys(InputFeature))
            )
    if "data" in config:
        experiments = pd.read_json(json.dumps(config["data"]), orient="split")

        for key in domain.get_feature_keys(OutputFeature):
            experiments[key] = 1
        domain.add_experiments(experiments=experiments)
    return domain
