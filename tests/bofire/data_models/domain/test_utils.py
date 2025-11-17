import numpy as np
import pandas as pd

from bofire.data_models.domain import api as domain_module
from bofire.data_models.domain.utils import LinearProjection
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

    # def test_create_qp_problem(
    #     self, mock_pymoo_generation: List[dict], repair_instance: LinearProjection
    # ):
    #     n_gen = len(mock_pymoo_generation)
    #     n_add = repair_instance.q
    #     d = repair_instance.d
    #     domain = repair_instance.domain
    #
    #     X = repair_instance.domain_handler.transform_mixed_to_2D(mock_pymoo_generation)
    #
    #     matrices = repair_instance._create_qp_problem_input(X)
    #
    #     P, q, G, h, A, b = (matrices.get(key) for key in ["P", "q", "G", "h", "A", "b"])
    #
    #     # numpy conversion
    #     P, G, A = (x.todense() for x in [P, G, A])
    #
    #     # check objective: x^T P x + q^T x
    #     assert (P == np.eye(n_gen * n_add * d)).all()
    #     assert (X.reshape(-1) == -q.reshape(-1)).all()
    #
    #     # box-bounds (upper part of G/h matrices)
    #     G_bounds = G[: 2 * n_gen * n_add * d, :]
    #     assert G_bounds.shape == (2 * n_gen * n_add * d, n_gen * n_add * d)
    #     h_bounds = h[: 2 * n_gen * n_add * d]
    #     assert len(h_bounds) == 2 * n_gen * n_add * d
    #     # assert structure of G
    #     assert (
    #         linalg.block_diag(
    #             *[np.vstack([np.eye(d), -np.eye(d)]) for _ in range(n_gen * n_add)]
    #         )
    #         == G_bounds
    #     ).all()
    #
    #     nck_constr = domain.constraints.get(includes=NChooseKConstraint).constraints
    #
    #     for xi in range(n_gen * n_add):
    #         ub = h_bounds[(xi * 2) * d : ((xi * 2) + 1) * d]
    #         lb = -h_bounds[((xi * 2) + 1) * d : ((xi * 2) + 2) * d]
    #
    #         if len(nck_constr) > 0:
    #             # check how NChooseK constraints manipulate the bounds
    #             for idx_constr in range(len(nck_constr)):
    #                 idx = repair_instance.n_choose_k_constr.idx[idx_constr]
    #                 lb, ub = lb[idx], ub[idx]
    #                 assert int((lb > 0).sum()) >= nck_constr[idx_constr].min_count
    #                 assert (
    #                     int((ub == 0).sum())
    #                     <= len(ub) - nck_constr[idx_constr].max_count
    #                 )
    #
    #         else:
    #             assert (
    #                 ub.reshape(-1) == repair_instance.bounds[1, :].numpy().reshape(-1)
    #             ).all()
    #             assert (
    #                 lb.reshape(-1) == repair_instance.bounds[0, :].numpy().reshape(-1)
    #             ).all()
    #
    #     # linear inequality constraints (lower part of G/h matrices)
    #     lin_ineq = domain.constraints.get(LinearInequalityConstraint)
    #     lin_ineq_coeffs = get_linear_constraints(domain, LinearInequalityConstraint)
    #     n_constr = len(lin_ineq.constraints)
    #     if n_constr > 0:
    #         G_constr = G[2 * n_gen * n_add * d :, :]
    #         h_constr = h[2 * n_gen * n_add * d :]
    #         assert G_constr.shape[0] == n_constr * n_add * n_gen
    #         assert G_constr.shape[1] == n_gen * n_add * d
    #         assert len(h_constr) == n_constr * n_add * n_gen
    #         for i, coeffs in enumerate(lin_ineq_coeffs):
    #             Gi_single = G_constr[i, :d]
    #             for idx_, val in zip(coeffs[0], coeffs[1]):
    #                 assert Gi_single[idx_] == -val
    #             assert h_constr[i] == -coeffs[2]
    #
    #     # linear equality constraints (A/b matrices)
    #     lin_eq = domain.constraints.get(LinearEqualityConstraint)
    #     lin_eq_coeffs = get_linear_constraints(domain, LinearEqualityConstraint)
    #     n_constr = len(lin_eq.constraints)
    #     if n_constr > 0:
    #         assert A.shape[0] == n_constr * n_add * n_gen
    #         assert A.shape[1] == n_gen * n_add * d
    #         assert len(b) == n_constr * n_add * n_gen
    #         for i, coeffs in enumerate(lin_eq_coeffs):
    #             Ai_single = A[i, :d]
    #             for idx_, val in zip(coeffs[0], coeffs[1]):
    #                 assert Ai_single[idx_] == -val
    #             assert b[i] == -coeffs[2]
