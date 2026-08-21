"""The normalizer must reduce every annotation shape in ``bofire.data_models``.

These tests deliberately assert against the *real* annotations rather than
synthetic ones, so that a refactor of a data model breaks them.
"""

import pytest

from bofire.data_models.constraints.api import Constraint
from bofire.data_models.domain.api import Constraints, Inputs, Outputs
from bofire.data_models.features.api import ContinuousInput
from bofire.data_models.introspection.annotations import normalize, type_tag
from bofire.data_models.strategies.api import SoboStrategy, StepwiseStrategy
from bofire.data_models.surrogates.api import BotorchSurrogates, SingleTaskGPSurrogate


def node_of(cls, field_name):
    field = cls.model_fields[field_name]
    return normalize(field.annotation, field)


def test_bare_tagged_union():
    node = node_of(SoboStrategy, "acquisition_function")
    assert node.kind == "union"
    assert node.discriminator == "type"
    assert "qUCB" in [type_tag(option) for option in node.options]


def test_optional_tagged_union_recovers_the_discriminator():
    """``FieldInfo.discriminator`` is not authoritative and must not be trusted alone.

    ``convergence_criterion`` is ``Optional[Annotated[Union[...], Field(discriminator=...)]]``
    and pydantic does not hoist the discriminator through ``Optional``. Reading
    the attribute alone renders this union as untagged.
    """
    assert SoboStrategy.model_fields["convergence_criterion"].discriminator is None

    node = node_of(SoboStrategy, "convergence_criterion")
    assert node.kind == "union"
    assert node.discriminator == "type"
    assert node.optional
    assert len(node.options) == 3


@pytest.mark.parametrize("container", [Inputs, Outputs])
def test_sequence_of_union_carries_the_discriminator_on_the_element(container):
    node = node_of(container, "features")
    assert node.kind == "list"
    assert node.element.kind == "union"
    assert node.element.discriminator == "type"


def test_typevar_bound_is_resolved_and_abstract_bases_are_separated():
    """``Constraints`` is a generic used unparametrized, so its element is ``~C``."""
    node = node_of(Constraints, "constraints")
    assert node.kind == "list"
    assert node.element.kind == "union"
    # The nested union is flattened and the abstract base is not offered as a choice.
    assert Constraint in node.element.abstract
    assert all(type_tag(option) for option in node.element.options)


def test_list_not_sequence_is_still_a_list():
    node = node_of(BotorchSurrogates, "surrogates")
    assert node.kind == "list"
    assert node.element.kind == "union"


def test_union_of_non_models_stays_a_leaf():
    """``bounds`` is ``Union[List[float], Tuple[float, float]]`` — nothing to navigate."""
    node = node_of(ContinuousInput, "bounds")
    assert node.kind == "scalar"
    assert "float" in node.display


def test_fixed_length_tuple_is_a_value_not_a_sequence():
    assert node_of(ContinuousInput, "bounds").kind == "scalar"


def test_optional_list_of_scalars():
    node = node_of(ContinuousInput, "local_relative_bounds")
    assert node.kind == "list"
    assert node.optional


def test_literal_discriminator():
    node = node_of(SoboStrategy, "type")
    assert node.kind == "literal"
    assert node.literal_values == ("SoboStrategy",)


def test_mapping():
    node = node_of(SingleTaskGPSurrogate, "categorical_encodings")
    assert node.kind == "mapping"
    assert node.element.kind == "union"


def test_enum():
    node = node_of(SingleTaskGPSurrogate, "output_scaler")
    assert node.kind == "enum"


def test_recursive_annotation_terminates():
    """Normalisation is bounded by annotation depth, not by graph reachability."""
    assert node_of(StepwiseStrategy, "steps").kind == "list"


def test_type_tag_is_none_for_classes_without_a_fixed_discriminator():
    # Abstract bases declare ``type: Any``; container models declare none at all.
    assert type_tag(Constraint) is None
    assert type_tag(BotorchSurrogates) is None
    assert type_tag(int) is None
