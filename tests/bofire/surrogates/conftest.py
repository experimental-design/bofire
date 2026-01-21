import pandas as pd
import pytest

from bofire.data_models.domain import api as domain_api
from bofire.data_models.features import api as features_api


@pytest.fixture
def chem_domain_simple() -> tuple[domain_api.Domain, pd.DataFrame, pd.DataFrame]:
    mols = ["CCC", "CCCC", "C(C)CC", "C(N)CC"]
    domain = domain_api.Domain(
        inputs=domain_api.Inputs(
            features=[
                features_api.CategoricalMolecularInput(key="molecules", categories=mols)
            ]
        ),
        outputs=domain_api.Outputs(
            features=[features_api.ContinuousOutput(key="output")]
        ),
    )
    X = pd.DataFrame({"molecules": mols})
    Y = pd.DataFrame({"output": [1.0, 2.0, 3.0, 4.0]})
    return domain, X, Y
