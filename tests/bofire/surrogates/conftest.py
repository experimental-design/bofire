import numpy as np
import pandas as pd
import pytest

from bofire.data_models.domain import api as domain_api
from bofire.data_models.features import api as features_api


@pytest.fixture
def chem_domain_simple(request) -> tuple[domain_api.Domain, pd.DataFrame, pd.DataFrame]:
    mols1 = ["CCC", "CCCC", "C(C)CC", "C(N)CC"]
    mols2 = [
        "NCC(=O)O",
        "CC(C(=O)O)N",
        "CC(C)C(C(=O)O)N",
        "CC(C(C(=O)O)N)O",
        "C1=CC=C(C=C1)CC(C(=O)O)N",
        "C1=CC=C2C(=C1)C(=CN2)CC(C(=O)O)N",
    ]

    multi_input: bool = request.param if hasattr(request, "param") else False

    domain = domain_api.Domain(
        inputs=domain_api.Inputs(
            features=[
                features_api.CategoricalMolecularInput(
                    key="molecules", categories=mols1
                )
            ]
        ),
        outputs=domain_api.Outputs(
            features=[features_api.ContinuousOutput(key="output")]
        ),
    )
    if multi_input:
        domain.inputs.features.append(
            features_api.CategoricalMolecularInput(key="molecules-2", categories=mols2)
        )

    if not multi_input:
        X = pd.DataFrame({"molecules": mols1})
        Y = pd.DataFrame({"output": [1.0, 2.0, 3.0, 4.0]})
    else:
        X = pd.DataFrame(
            {
                "molecules": np.random.choice(mols1, (20,)),
                "molecules-2": np.random.choice(mols2, (20,)),
            }
        )
        Y = pd.DataFrame({"output": np.random.uniform(high=4.0, size=(20,))})

    return domain, X, Y
