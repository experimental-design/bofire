from copy import deepcopy
from typing import List

from bofire.data_models.domain.api import Inputs, Outputs
from bofire.data_models.domain.domain import Domain
from bofire.data_models.features.api import Input


def get_subdomain(
    domain: Domain,
    feature_keys: List,
) -> Domain:
    """Removes all features not defined as argument creating a subdomain of the provided domain

    Args:
        domain (Domain): the original domain wherefrom a subdomain should be created
        feature_keys (List): List of features that shall be included in the subdomain

    Raises:
        Assert: when in total less than 2 features are provided
        ValueError: when a provided feature key is not present in the provided domain
        Assert: when no output feature is provided
        Assert: when no input feature is provided
        ValueError: _description_

    Returns:
        Domain: A new domain containing only parts of the original domain

    """
    assert len(feature_keys) >= 2, "At least two features have to be provided."
    outputs = []
    inputs = []
    for key in feature_keys:
        try:
            feat = (domain.inputs + domain.outputs).get_by_key(key)
        except KeyError:
            raise ValueError(f"Feature {key} not present in domain.")
        if isinstance(feat, Input):
            inputs.append(feat)
        else:
            outputs.append(feat)
    assert len(outputs) > 0, "At least one output feature has to be provided."
    assert len(inputs) > 0, "At least one input feature has to be provided."
    inputs = Inputs(features=inputs)
    outputs = Outputs(features=outputs)
    # loop over constraints and make sure that all features used in constraints are in the input_feature_keys
    for c in domain.constraints:
        for key in c.features:
            if key not in inputs.get_keys():
                raise ValueError(
                    f"Removed input feature {key} is used in a constraint.",
                )
    subdomain = deepcopy(domain)
    subdomain.inputs = inputs
    subdomain.outputs = outputs
    return subdomain
