import importlib

import pytest

import bofire.data_models.molfeatures.names as names

RDKIT_AVAILABLE = importlib.util.find_spec("rdkit") is not None


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
def test_fragments():
    from rdkit.Chem import Descriptors

    fragment_list = [
        item[0] for item in Descriptors.descList if item[0].startswith("fr_")
    ]

    assert names.fragments == fragment_list


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires rdkit")
def test_mordred():
    from mordred import Calculator
    from mordred import descriptors as mordred_descriptors

    calc = Calculator(mordred_descriptors, ignore_3D=False)
    assert names.mordred == [str(d) for d in calc.descriptors]
