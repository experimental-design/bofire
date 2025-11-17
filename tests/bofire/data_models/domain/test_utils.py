import numpy as np
import pandas as pd

from bofire.data_models.domain import api as domain_module
from bofire.data_models.features import api as feature_module

from bofire.data_models.domain.utils import LinearProjection

class TestLinearProjection:

    def test_feature_ordering(self):
        """ check for mismatches in re-ordering features"""


        domain = domain_module.Domain(
            inputs=domain_module.Inputs(features=[
                feature_module.CategoricalInput(
                    key="cat feature", categories=["A", "B", "C"], allowed=[True, True, False],
                ),
                feature_module.ContinuousInput(
                    key="Z-feature", bounds=(0., 1.)
                ),
                feature_module.ContinuousInput(
                    key="A-feature", bounds=(1., 2.)
                ),
            ])
        )

        linear_projection = LinearProjection(domain)


        # test examples
        experiments = pd.DataFrame({
            "Z-feature": 100,
            "cat feature": "A",
            "A-feature": 100.,

        }, index=[0])

        corrected_experiments = linear_projection(experiments)
        assert np.allclose(corrected_experiments["A-feature"], np.array([2.]))
        assert np.allclose(corrected_experiments["Z-feature"], np.array([1.]))

