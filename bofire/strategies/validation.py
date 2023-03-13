from bofire.data_models.domain.api import Domain
from bofire.data_models.objectives.api import Objective


def validate_constraints(cls, domain: Domain):
    """Validator to ensure that all constraints defined in the domain are valid for the chosen strategy

    Args:
        domain (Domain): The domain to be used in the strategy

    Raises:
        ValueError: if a constraint is defined in the domain but is invalid for the strategy chosen

    Returns:
        Domain: the domain
    """
    for constraint in domain.constraints:
        if not cls.is_constraint_implemented(type(constraint)):
            raise ValueError(
                f"constraint `{type(constraint)}` is not implemented for strategy `{cls.__name__}`"  # type: ignore
            )
    return domain


def validate_features(cls, domain: Domain):
    """Validator to ensure that all features defined in the domain are valid for the chosen strategy

    Args:
        domain (Domain): The domain to be used in the strategy

    Raises:
        ValueError: if a feature type is defined in the domain but is invalid for the strategy chosen

    Returns:
        Domain: the domain
    """
    for feature in domain.inputs + domain.output_features:
        if not cls.is_feature_implemented(type(feature)):
            raise ValueError(
                f"feature `{type(feature)}` is not implemented for strategy `{cls.__name__}`"  # type: ignore
            )
    return domain


def validate_input_feature_count(cls, domain: Domain):
    """Validator to ensure that at least one input is defined.

    Args:
        domain (Domain): The domain to be used in the strategy

    Raises:
        ValueError: if no input feature is specified

    Returns:
        Domain: the domain
    """
    if len(domain.input_features) == 0:
        raise ValueError("no input feature specified")
    return domain


def validate_output_feature_count(cls, domain: Domain):
    """Validator to ensure that at least one output feature with attached objective is defined.

    Args:
        domain (Domain): The domain to be used in the strategy

    Raises:
        ValueError: if no output feature is specified
        ValueError: if not output feauture with an attached objective is specified

    Returns:
        Domain: the domain
    """
    if len(domain.output_features) == 0:
        raise ValueError("no output feature specified")
    if len(domain.outputs.get_by_objective(Objective)) == 0:
        raise ValueError("no output feature with objective specified")
    return domain
