from copy import deepcopy
from typing import List, Tuple, cast

import numpy as np
import pandas as pd

from bofire.data_models.constraints.api import (
    AnyConstraint,
    Constraint,
    LinearConstraint,
    LinearEqualityConstraint,
    LinearInequalityConstraint,
    NChooseKConstraint,
)
from bofire.data_models.domain.api import Constraints, Domain, Inputs
from bofire.data_models.features.api import ContinuousInput, Input


### this module is based on the original implementation in basf/opti.


class AffineTransform:
    """Class to switch back and forth from the reduced to the original domain."""

    def __init__(self, equalities: List[Tuple[str, List[str], List[float]]]):
        """Initializes a `AffineTransformation` object.

        Args:
            equalities (List[Tuple[str,List[str],List[float]]]): List of equalities. Every equality
                is defined as a tuple, in which the first entry is the key of the reduced feature, the second
                one is a list of feature keys that can be used to compute the feature and the third list of floats
                are the corresponding coefficients.

        """
        self.equalities = equalities

    def augment_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Restore the eliminated features in a dataframe

        Args:
            data (pd.DataFrame): Dataframe that should be restored.

        Returns:
            pd.DataFrame: Restored dataframe

        """
        if len(self.equalities) == 0:
            return data
        data = data.copy()
        for name_lhs, names_rhs, coeffs in self.equalities:
            data[name_lhs] = coeffs[-1]
            for i, name in enumerate(names_rhs):
                data[name_lhs] += coeffs[i] * data[name]
        return data

    def drop_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Drop eliminated features from a dataframe.

        Args:
            data (pd.DataFrame): Dataframe with features to be dropped.

        Returns:
            pd.DataFrame: Reduced dataframe.

        """
        if len(self.equalities) == 0:
            return data
        drop = []
        for name_lhs, _, _ in self.equalities:
            if name_lhs in data.columns:
                drop.append(name_lhs)
        return data.drop(columns=drop)


def reduce_domain(domain: Domain) -> Tuple[Domain, AffineTransform]:
    """Reduce a domain with linear equality constraints to a subdomain where linear equality constraints are eliminated.

    Args:
        domain (Domain): Domain to be reduced.

    Returns:
        Tuple[Domain, AffineTransform]: reduced domain and the according transformation to switch between the
            reduced and original domain.

    """
    # check if the domain can be reduced
    if not check_domain_for_reduction(domain):
        return domain, AffineTransform([])

    # find linear equality constraints
    linear_equalities = domain.constraints.get(LinearEqualityConstraint)
    other_constraints = domain.constraints.get(
        Constraint,
        excludes=[LinearEqualityConstraint],
    )

    # only consider continuous inputs
    continuous_inputs = [
        cast(ContinuousInput, f) for f in domain.inputs.get(ContinuousInput)
    ]
    other_inputs = domain.inputs.get(Input, excludes=[ContinuousInput])

    # assemble Matrix A from equality constraints
    N = len(linear_equalities)
    M = len(continuous_inputs) + 1
    names = np.concatenate(([feat.key for feat in continuous_inputs], ["rhs"]))

    A_aug = pd.DataFrame(data=np.zeros(shape=(N, M)), columns=names)

    for i in range(len(linear_equalities)):
        c = linear_equalities[i]
        assert isinstance(c, LinearEqualityConstraint)
        A_aug.loc[i, c.features] = c.coefficients
        A_aug.loc[i, "rhs"] = c.rhs
    A_aug = A_aug.values

    # catch special cases
    check_existence_of_solution(A_aug)

    # bring A_aug to reduced row-echelon form
    A_aug_rref, pivots = rref(A_aug)
    pivots = np.array(pivots)
    A_aug_rref = np.array(A_aug_rref).astype(np.float64)

    # formulate box bounds as linear inequality constraints in matrix form
    B = np.zeros(shape=(2 * (M - 1), M))
    B[: M - 1, : M - 1] = np.eye(M - 1)
    B[M - 1 :, : M - 1] = -np.eye(M - 1)

    B[: M - 1, -1] = np.array([feat.upper_bound for feat in continuous_inputs])
    B[M - 1 :, -1] = -1.0 * np.array([feat.lower_bound for feat in continuous_inputs])

    # eliminate columns with pivot element
    for i in range(len(pivots)):
        p = pivots[i]
        B[p, :] -= A_aug_rref[i, :]
        B[p + M - 1, :] += A_aug_rref[i, :]

    # build up reduced domain
    _domain = Domain.model_construct(
        # _fields_set = {"inputs", "outputs", "constraints"}
        inputs=deepcopy(other_inputs),
        outputs=deepcopy(domain.outputs),
        constraints=deepcopy(other_constraints),
    )
    new_inputs = [
        deepcopy(feat) for i, feat in enumerate(continuous_inputs) if i not in pivots
    ]
    all_inputs = _domain.inputs + new_inputs
    assert isinstance(all_inputs, Inputs)
    _domain.inputs.features = all_inputs.features

    constraints: List[AnyConstraint] = []
    for i in pivots:
        # reduce equation system of upper bounds
        ind = np.where(B[i, :-1] != 0)[0]
        if len(ind) > 0 and B[i, -1] < np.inf:
            if len(list(names[ind])) > 1:
                c = LinearInequalityConstraint.from_greater_equal(
                    features=list(names[ind]),
                    coefficients=(-1.0 * B[i, ind]).tolist(),
                    rhs=B[i, -1] * -1.0,
                )
                constraints.append(c)
            else:
                key = names[ind][0]
                feat = cast(ContinuousInput, _domain.inputs.get_by_key(key))
                adjust_boundary(feat, (-1.0 * B[i, ind])[0], B[i, -1] * -1.0)
        elif B[i, -1] < -1e-16:
            raise Exception("There is no solution that fulfills the constraints.")

        # reduce equation system of lower bounds
        ind = np.where(B[i + M - 1, :-1] != 0)[0]
        if len(ind) > 0 and B[i + M - 1, -1] < np.inf:
            if len(list(names[ind])) > 1:
                c = LinearInequalityConstraint.from_greater_equal(
                    features=list(names[ind]),
                    coefficients=(-1.0 * B[i + M - 1, ind]).tolist(),
                    rhs=B[i + M - 1, -1] * -1.0,
                )
                constraints.append(c)
            else:
                key = names[ind][0]
                feat = cast(ContinuousInput, _domain.inputs.get_by_key(key))
                adjust_boundary(
                    feat,
                    (-1.0 * B[i + M - 1, ind])[0],
                    B[i + M - 1, -1] * -1.0,
                )
        elif B[i + M - 1, -1] < -1e-16:
            raise Exception("There is no solution that fulfills the constraints.")

    if len(constraints) > 0:
        _domain.constraints.constraints = _domain.constraints.constraints + constraints  # type: ignore

    # assemble equalities
    _equalities = []
    for i in range(len(pivots)):
        name_lhs = names[pivots[i]]
        names_rhs = []
        coeffs = []

        for j in range(len(names) - 1):
            if A_aug_rref[i, j] != 0 and j != pivots[i]:
                coeffs.append(-A_aug_rref[i, j])
                names_rhs.append(names[j])

        coeffs.append(A_aug_rref[i, -1])

        _equalities.append((name_lhs, names_rhs, coeffs))

    trafo = AffineTransform(_equalities)
    # remove remaining dependencies of eliminated inputs from the problem
    _domain = remove_eliminated_inputs(_domain, trafo)
    return _domain, trafo


def check_domain_for_reduction(domain: Domain) -> bool:
    """Check if the reduction can be applied or if a trivial case is present.

    Args:
        domain (Domain): Domain to be checked.

    Returns:
        bool: True if reducable, else False.

    """
    # are there any constraints?
    if len(domain.constraints) == 0:
        return False

    # are there any linear equality constraints?
    linear_equalities = domain.constraints.get(LinearEqualityConstraint)
    if len(linear_equalities) == 0:
        return False

    # are there no NChooseKConstraint constraints?
    if len(domain.constraints.get([NChooseKConstraint])) > 0:
        return False

    # are there continuous inputs
    continuous_inputs = domain.inputs.get(ContinuousInput)
    if len(continuous_inputs) == 0:
        return False

    # check that equality constraints only contain continuous inputs
    for c in linear_equalities:
        assert isinstance(c, LinearConstraint)
        for feat in c.features:
            if feat not in domain.inputs.get_keys(ContinuousInput):
                return False
    return True


def check_existence_of_solution(A_aug):
    """Given an augmented coefficient matrix this function determines the existence (and uniqueness) of solution using the rank theorem."""
    A = A_aug[:, :-1]
    b = A_aug[:, -1]
    len_inputs = np.shape(A)[1]

    # catch special cases
    rk_A_aug = np.linalg.matrix_rank(A_aug)
    rk_A = np.linalg.matrix_rank(A)

    if rk_A == rk_A_aug:
        if rk_A < len_inputs:
            return  # all good
        x = np.linalg.solve(A, b)
        raise Exception(
            f"There is a unique solution x for the linear equality constraints: x={x}",
        )
    if rk_A < rk_A_aug:
        raise Exception(
            "There is no solution fulfilling the linear equality constraints.",
        )


def remove_eliminated_inputs(domain: Domain, transform: AffineTransform) -> Domain:
    """Eliminates remaining occurrences of eliminated inputs in linear constraints.

    Args:
        domain (Domain): Domain in which the linear constraints should be purged.
        transform (AffineTransform): Affine transformation object that defines the obsolete features.

    Raises:
        ValueError: If feature occurs in a constraint different from a linear one.

    Returns:
        Domain: Purged domain.

    """
    inputs_names = domain.inputs.get_keys()
    M = len(inputs_names)

    # write the equalities for the backtransformation into one matrix
    inputs_dict = {inputs_names[i]: i for i in range(M)}

    # build up dict from domain.equalities e.g. {"xi1": [coeff(xj1), ..., coeff(xjn)], ... "xik":...}
    coeffs_dict = {}
    for e in transform.equalities:
        coeffs = np.zeros(M + 1)
        for j, name in enumerate(e[1]):
            coeffs[inputs_dict[name]] = e[2][j]
        coeffs[-1] = e[2][-1]
        coeffs_dict[e[0]] = coeffs

    constraints = []
    for c in domain.constraints.get():
        # Nonlinear constraints not supported
        if not isinstance(c, LinearConstraint):
            raise ValueError(
                "Elimination of variables is only supported for LinearEquality and LinearInequality constraints.",
            )

        # no changes, if the constraint does not contain eliminated inputs
        if all(name in inputs_names for name in c.features):
            constraints.append(c)

        # remove inputs from the constraint that were eliminated from the inputs before
        else:
            totally_removed = False
            _features = np.array(inputs_names)
            _rhs = c.rhs

            # create new lhs and rhs from the old one and knowledge from problem._equalities
            _coefficients = np.zeros(M)
            for j, name in enumerate(c.features):
                if name in inputs_names:
                    _coefficients[inputs_dict[name]] += c.coefficients[j]
                else:
                    _coefficients += c.coefficients[j] * coeffs_dict[name][:-1]
                    _rhs -= c.coefficients[j] * coeffs_dict[name][-1]

            _features = _features[np.abs(_coefficients) > 1e-16]
            _coefficients = _coefficients[np.abs(_coefficients) > 1e-16]
            _c = None
            if isinstance(c, LinearEqualityConstraint):
                if len(_features) > 1:
                    _c = LinearEqualityConstraint(
                        features=_features.tolist(),
                        coefficients=_coefficients.tolist(),
                        rhs=_rhs,
                    )
                elif len(_features) == 0:
                    totally_removed = True
                else:
                    feat: ContinuousInput = ContinuousInput(
                        **domain.inputs.get_by_key(_features[0]).model_dump(),
                    )
                    feat.bounds = [_coefficients[0], _coefficients[0]]
                    totally_removed = True
            elif len(_features) > 1:
                _c = LinearInequalityConstraint(
                    features=_features.tolist(),
                    coefficients=_coefficients.tolist(),
                    rhs=_rhs,
                )
            elif len(_features) == 0:
                totally_removed = True
            else:
                feat = cast(ContinuousInput, domain.inputs.get_by_key(_features[0]))
                adjust_boundary(feat, _coefficients[0], _rhs)
                totally_removed = True

            # check if constraint is always fulfilled/not fulfilled
            if not totally_removed:
                assert _c is not None
                if len(_c.features) == 0 and _c.rhs >= 0:
                    pass
                elif len(_c.features) == 0 and _c.rhs < 0:
                    raise Exception("Linear constraints cannot be fulfilled.")
                elif np.isinf(_c.rhs):
                    pass
                else:
                    constraints.append(_c)
    domain.constraints = Constraints(constraints=constraints)
    return domain


def rref(A: np.ndarray, tol: float = 1e-8) -> Tuple[np.ndarray, List[int]]:
    """Computes the reduced row echelon form of a Matrix

    Args:
        A (ndarray): 2d array representing a matrix.
        tol (float, optional): tolerance for rounding to 0. Defaults to 1e-8.

    Returns:
        (A_rref, pivots), where A_rref is the reduced row echelon form of A and pivots
        is a numpy array containing the pivot columns of A_rref

    """
    A = np.array(A, dtype=np.float64)
    n, m = np.shape(A)

    col = 0
    row = 0
    pivots = []

    for col in range(m):
        # does a pivot element exist?
        if all(np.abs(A[row:, col]) < tol):
            pass
        # if yes: start elimination
        else:
            pivots.append(col)
            max_row = np.argmax(np.abs(A[row:, col])) + row
            # switch to most stable row
            A[[row, max_row], :] = A[[max_row, row], :]
            # normalize row
            A[row, :] /= A[row, col]
            # eliminate other elements from column
            for r in range(n):
                if r != row:
                    A[r, :] -= A[r, col] / A[row, col] * A[row, :]
            row += 1

    prec = int(-np.log10(tol))
    return np.round(A, prec), pivots


def adjust_boundary(feature: ContinuousInput, coef: float, rhs: float):
    """Adjusts the boundaries of a feature.

    Args:
        feature (ContinuousInput): Feature to be adjusted.
        coef (float): Coefficient.
        rhs (float): Right-hand-side of the constraint.

    """
    boundary = rhs / coef
    if coef > 0:
        if boundary > feature.lower_bound:
            feature.bounds = [boundary, feature.upper_bound]
    elif boundary < feature.upper_bound:
        feature.bounds = [feature.lower_bound, boundary]
