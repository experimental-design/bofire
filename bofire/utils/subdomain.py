from copy import deepcopy
from typing import List

from bofire.data_models.domain.api import Inputs
from bofire.data_models.domain.domain import Domain
from bofire.data_models.features.api import Input


def get_subdomain(
    domain: Domain,
    feature_keys: List,
) -> Domain:
    """removes all features not defined as argument creating a subdomain of the provided domain

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
    output_features = []
    input_features = []
    for key in feature_keys:
        try:
            feat = domain.get_feature(key)
        except KeyError:
            raise ValueError(f"Feature {key} not present in domain.")
        if isinstance(feat, Input):
            input_features.append(feat)
        else:
            output_features.append(feat)
    assert len(output_features) > 0, "At least one output feature has to be provided."
    assert len(input_features) > 0, "At least one input feature has to be provided."
    input_features = Inputs(features=input_features)
    # loop over constraints and make sure that all features used in constraints are in the input_feature_keys
    for c in domain.constraints:
        # TODO: fix type hint
        for key in c.features:  # type: ignore
            if key not in input_features.get_keys():
                raise ValueError(
                    f"Removed input feature {key} is used in a constraint."
                )
    subdomain = deepcopy(domain)
    subdomain.input_features = input_features
    subdomain.output_features = output_features
    return subdomain
