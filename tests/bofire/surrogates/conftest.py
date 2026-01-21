import pytest

import pandas as pd
from bofire.data_models.domain import api as domain_api
from bofire.data_models.features import api as features_api


@pytest.fixture
def chem_domain_simple() -> tuple[domain_api.Domain, pd.DataFrame, pd.DataFrame]:
    domain = domain_api.Domain(
        inputs=domain_api.Inputs(
            features=[
                features_api.CategoricalMolecularInput(
                    key="molecules", categories=["C(O)O", "O", "CC"]
                )
            ]
        ),
        outputs=domain_api.Outputs(
            features=[features_api.ContinuousOutput(key="output")]
        ),
    )
    X = pd.DataFrame({"molecules": ["C(O)O", "O", "CC"]})
    Y = pd.DataFrame({"output": [1.0, 2.0, 3.0]})
    return domain, X, Y
