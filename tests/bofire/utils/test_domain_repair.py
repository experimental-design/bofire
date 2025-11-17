import numpy as np
import pandas as pd

from bofire.data_models.domain import api as domain_module
from bofire.utils.domain_repair import LinearProjection
from bofire.data_models.features import api as feature_module


class TestLinearProjection:
    def test_feature_ordering(self):
        """check for mismatches in re-ordering features"""

        domain = domain_module.Domain(
            inputs=domain_module.Inputs(
                features=[
                    feature_module.CategoricalInput(
                        key="cat feature",
                        categories=["A", "B", "C"],
                        allowed=[True, True, False],
                    ),
                    feature_module.ContinuousInput(key="Z-feature", bounds=(0.0, 1.0)),
                    feature_module.ContinuousInput(key="A-feature", bounds=(1.0, 2.0)),
                ]
            )
        )

        linear_projection = LinearProjection(domain)

        # test examples
        experiments = pd.DataFrame(
            {
                "Z-feature": 100,
                "cat feature": "A",
                "A-feature": 100.0,
            },
            index=[0],
        )

        # a) using the pandas-based __call__ method
        corrected_experiments = linear_projection(experiments)
        assert np.allclose(corrected_experiments["A-feature"], np.array([2.0]))
        assert np.allclose(corrected_experiments["Z-feature"], np.array([1.0]))

        # b) using the low-level matrix method
        X = np.zeros((1, 3))
        X_corrected = linear_projection.solve_numeric(X)
        assert (
            X_corrected[
                0,
                domain.inputs.get_feature_indices(
                    linear_projection.input_preprocessing_specs, ["A-feature"]
                )[0],
            ]
            == 1.0
        )