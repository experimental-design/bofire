import json
from typing import Dict, Optional

import numpy as np
import pandas as pd

from bofire.domain import Domain
from bofire.domain.constraints import (
    Constraint,
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
    Feature,
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


def feature2input(feature: InputFeature):
    if isinstance(feature, ContinuousInputFeature):
        return {
            "type": "continuous",
            "name": feature.key,
            "domain": [feature.lower_bound, feature.upper_bound],
        }
    if isinstance(feature, DiscreteInputFeature):
        return {
            "type": "discrete",
            "name": feature.key,
            "domain": feature.values,
        }
    if isinstance(feature, CategoricalInputFeature):
        return {
            "type": "categorical",
            "name": feature.key,
            "domain": feature.get_allowed_categories(),
        }
    else:
        raise ValueError(f"Unsupported feature type {feature.__class__.__name__}.")


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


def feature2objective(feature: OutputFeature):
    assert hasattr(
        feature, "desirability_function"
    ), "Feature has no attribute `desirability_function`"
    if isinstance(feature.desirability_function, MinIdentityDesirabilityFunction):
        return {
            "name": feature.key,
            "type": "minimize",
        }
    if isinstance(feature.desirability_function, MaxIdentityDesirabilityFunction):
        return {
            "name": feature.key,
            "type": "maximize",
        }
    if isinstance(feature.desirability_function, CloseToTargetDesirabilityFunction):
        d = {
            "name": feature.key,
            "type": "close-to-target",
            # "exponent": feature.desirability_function.exponent,
            "target": feature.desirability_function.target_value,
            # "tolerance": feature.desirability_function.tolerance,
        }
        if feature.desirability_function.exponent != 1:
            d["exponent"] = feature.desirability_function.exponent
        if feature.desirability_function.tolerance != 0:
            d["tolerance"] = feature.desirability_function.tolerance
        return d
    else:
        raise ValueError(f"Unsupported feature type {feature.__class__.__name__}.")


def opti_constraint2constraint(config: Dict, input_feature_keys: Optional[list] = None):
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
    if config["type"] == "nonlinear-equality":
        return NonlinearEqualityConstraint(
            features=input_feature_keys, expression=config["expression"]
        )
    if config["type"] == "nonlinear-inequality":
        return NonlinearInqualityConstraint(
            features=input_feature_keys, expression=config["expression"]
        )
    raise ValueError(f"Unknown constraint type {config['type']}.")


def constraint2opti_constraint(constraint: Constraint):
    if isinstance(constraint, LinearEqualityConstraint):
        return {
            "type": "linear-equality",
            "names": constraint.features,
            "lhs": constraint.coefficients,
            "rhs": constraint.rhs,
        }
    if isinstance(constraint, LinearInequalityConstraint):
        return {
            "type": "linear-inequality",
            "names": constraint.features,
            "lhs": constraint.coefficients,
            "rhs": constraint.rhs,
        }
    if isinstance(constraint, NChooseKConstraint):
        if constraint.min_count > 0:
            raise ValueError("min_count > 0 not supported in opt.")
        if constraint.none_also_valid:
            raise ValueError("none_also_valid == True not supported in opti.")
        return {
            "type": "n-choose-k",
            "names": constraint.features,
            "max_active": constraint.max_count,
        }
    if isinstance(constraint, NonlinearEqualityConstraint):
        return {
            "type": "nonlinear-equality",
            "expression": constraint.expression,
        }
    if isinstance(constraint, NonlinearInqualityConstraint):
        return {
            "type": "nonlinear-inequality",
            "expression": constraint.expression,
        }
    else:
        raise ValueError(
            f"Unsupported constraint type {constraint.__class__.__name__}."
        )


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
                opti_constraint2constraint(
                    cconfig, domain.get_feature_keys(InputFeature)
                )
            )
    if "data" in config:
        experiments = pd.read_json(json.dumps(config["data"]), orient="split")

        for key in domain.get_feature_keys(OutputFeature):
            experiments[f"valid_{key}"] = 1
        domain.add_experiments(experiments=experiments)
    return domain


def domain2problem(domain: Domain, name: Optional[str] = None) -> Dict:
    config = {
        "name": name,
        "inputs": [feature2input(feat) for feat in domain.input_features],
        "constraints": [
            constraint2opti_constraint(constraint) for constraint in domain.constraints
        ],
        "objectives": [feature2objective(feat) for feat in domain.output_features],
        "outputs": [
            {"type": "continuous", "name": feat.key} for feat in domain.output_features
        ],
    }
    if domain.experiments is not None:
        config["data"] = (
            domain.experiments[domain.get_feature_keys(Feature)]
            .replace({np.nan: None})
            .to_dict("split")
        )
    return config
