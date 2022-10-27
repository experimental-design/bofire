from typing import Dict

from everest.domain import Domain
from everest.domain.features import (
    CategoricalInputFeature,
    ContinuousInputFeature,
    DiscreteInputFeature,
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


def output2feature(parameter_config: Dict, objective_config: Dict):
    pass


def constraint2constraint(constraint_config: Dict):
    pass


def problem2domain(config: Dict):
    input_features = []
    output_features = []
    for pconfig in config["inputs"]:
        input_features.append(input2feature(pconfig))
    for pconfig, oconfig in zip(config["outputs"], config["objectives"]):
        output_features.append(output2feature(pconfig, oconfig))
    domain = Domain(input_features=input_features, output_features=output_features)
    if "constraints" in config:
        for cconfig in config["constraints"]:
            domain.add_constraint(constraint2constraint(cconfig))
