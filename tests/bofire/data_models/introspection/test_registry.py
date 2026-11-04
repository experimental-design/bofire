import subprocess
import sys

import pytest

from bofire.data_models.api import AnyThing
from bofire.data_models.introspection import registry
from bofire.data_models.introspection.registry import (
    AmbiguousNameError,
    UnknownNameError,
    ambiguous_names,
    lookup,
    names,
)
from bofire.data_models.strategies.api import GeneticAlgorithmOptimizer


def test_lookup_by_class_name():
    assert lookup("qUCB").__name__ == "qUCB"


def test_lookup_by_type_tag():
    """The tag is what a caller writes into JSON, and it is not always the class name."""
    assert GeneticAlgorithmOptimizer.model_fields["type"].default == "GeneticAlgorithm"
    assert lookup("GeneticAlgorithm") is GeneticAlgorithmOptimizer
    assert lookup("GeneticAlgorithmOptimizer") is GeneticAlgorithmOptimizer


def test_lookup_by_qualified_name():
    assert lookup("bofire.data_models.acquisition_functions.acquisition_function.qUCB")


@pytest.mark.parametrize(
    "name",
    ["BotorchSurrogates", "Step", "BotorchOptimizer", "ObjectiveImprovementCriterion"],
)
def test_registry_covers_types_missing_from_anything(name):
    """``AnyThing`` is assembled from a few unions and omits reachable types."""
    assert name not in {cls.__name__ for cls in AnyThing}
    assert lookup(name)


def test_registry_is_a_superset_of_anything():
    assert {cls.__name__ for cls in AnyThing} <= names()


def test_ambiguous_names_are_exactly_the_known_collision():
    """A ratchet: if this fails, a new duplicate class name was introduced.

    Add a fully qualified name at the call site (or rename the class) rather than
    loosening this test -- an ambiguous name cannot be resolved by any rule.
    """
    assert ambiguous_names() == {"Condition"}


def test_ambiguous_lookup_raises_with_qualified_alternatives():
    with pytest.raises(AmbiguousNameError, match="fully qualified"):
        lookup("Condition")


def test_unknown_name_raises():
    with pytest.raises(UnknownNameError):
        lookup("NoSuchDataModel")


def test_private_and_parametrized_generics_are_not_indexed():
    assert "_BaseFeatures" not in names()
    assert not [name for name in names() if "[" in name]


def test_importing_the_package_neither_walks_nor_imports_torch():
    """The walk imports every data model module, so it must stay lazy.

    Checked in a subprocess: within this session other tests have already
    imported plenty, so ``sys.modules`` here says nothing. ``torch`` is absent
    altogether in the ``testing_data_models`` CI job.
    """
    source = (
        "import sys\n"
        "import bofire.data_models.introspection.api as api\n"
        "from bofire.data_models.introspection import registry\n"
        "assert registry._index.cache_info().misses == 0, 'walked at import time'\n"
        "assert 'torch' not in sys.modules, 'torch imported at import time'\n"
    )
    subprocess.run([sys.executable, "-c", source], check=True)


def test_the_walk_is_cached():
    registry._index()
    before = registry._index.cache_info()
    registry._index()
    assert registry._index.cache_info().misses == before.misses
