import numpy as np
import pandas as pd
import pytest
from cvxpy.error import SolverError

from bofire.data_models.constraints import api as constraing_module
from bofire.data_models.domain import api as domain_module
from bofire.data_models.features import api as feature_module
from bofire.utils.domain_repair import LinearProjection


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

    def test_scaling(self):
        domain = domain_module.Domain(
            inputs=domain_module.Inputs(
                features=[
                    feature_module.ContinuousInput(key="small", bounds=(1.0, 2.0)),
                    feature_module.ContinuousInput(key="large", bounds=(1e8, 1e9)),
                ]
            ),
            constraints=domain_module.Constraints(
                constraints=[
                    constraing_module.LinearEqualityConstraint(  # a diagonal line in scaled small/large space
                        features=["small", "large"],
                        coefficients=[-9e8, 1.0],
                        rhs=-8e8,
                    )
                ]
            ),
        )

        linear_projection_no_scaling = LinearProjection(domain, scale_problem=False)
        linear_projection_with_scaling = LinearProjection(domain, scale_problem=True)

        # sample in complete space
        experiments = pd.DataFrame(
            {
                "small": np.random.uniform(low=1.0, high=2.0, size=(100,)),
                "large": np.random.uniform(low=1e8, high=1e9, size=(100,)),
            }
        )

        # project to diagonal with/without scaling
        with pytest.raises(
            SolverError
        ):  # this fails, due to badly conditioned constraints
            _ = linear_projection_no_scaling(experiments)

        # this works, and projects on the diagonal
        corrected_with_scaling = linear_projection_with_scaling(experiments)

        assert domain.constraints.is_fulfilled(corrected_with_scaling).all()
