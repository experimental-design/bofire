import inspect
import warnings
from typing import TYPE_CHECKING, Callable, Dict, Literal, Optional, Union

import numpy as np
import pandas as pd
from pydantic import Field, field_validator, model_validator


try:
    import torch
    from torch.autograd.functional import hessian as torch_hessian
    from torch.autograd.functional import jacobian as torch_jacobian

    torch_tensor = torch.tensor
    torch_diag = torch.diag
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

    def error_func(*args, **kwargs):
        raise NotImplementedError("torch must be installed to use this functionality")

    torch_jacobian = error_func  # ty: ignore[invalid-assignment]
    torch_tensor = error_func
    torch_diag = error_func
    torch_hessian = error_func  # ty: ignore[invalid-assignment]

if TYPE_CHECKING:  # pragma: no cover
    import torch as _torch

from bofire.data_models.constraints.constraint import (
    EqualityConstraint,
    InequalityConstraint,
    IntrapointConstraint,
)
from bofire.data_models.domain.features import Inputs
from bofire.data_models.features.api import ContinuousInput


class NonlinearConstraint(IntrapointConstraint):
    """Base class for nonlinear equality and inequality constraints.

    Attributes:
        expression (str): Mathematical expression that can be evaluated by `pandas.eval`.
        jacobian_expression (str): Mathematical expression that that can be evaluated by `pandas.eval`.
        features (list): list of feature keys (str) on which the constraint works on.

    """

    expression: Union[str, Callable]
    jacobian_expression: Optional[Union[str, Callable]] = Field(
        default=None, validate_default=True
    )
    hessian_expression: Optional[Union[str, Callable]] = Field(
        default=None, validate_default=True
    )

    def validate_inputs(self, inputs: Inputs):
        """Validate that all constraint features are continuous inputs.
        Args:
            inputs (Inputs): Input feature collection from the domain.
        Raises:
            ValueError: If any feature is not a ContinuousInput.
        """
        keys = inputs.get_keys(ContinuousInput)
        for f in self.features:
            if f not in keys:
                raise ValueError(
                    f"Feature {f} is not a continuous input feature in the provided Inputs object.",
                )

    @model_validator(mode="after")
    def validate_features(self):
        """Validate that provided features match callable expression arguments."""
        if isinstance(self.expression, Callable):
            features = list(inspect.getfullargspec(self.expression).args)
            if set(features) != set(self.features):
                raise ValueError(
                    "Provided features do not match the features used in the expression.",
                )
        return self

    @field_validator("jacobian_expression")
    @classmethod
    def set_jacobian_expression(cls, jacobian_expression, info) -> Union[str, Callable]:
        """Auto-compute Jacobian using SymPy for string expressions if not provided.
        Args:
            jacobian_expression: User-provided Jacobian or None.
            info: Pydantic validation context.
        Returns:
            Union[str, Callable]: Jacobian expression.
        """
        if (
            jacobian_expression is None
            and "features" in info.data.keys()
            and "expression" in info.data.keys()
        ):
            try:
                import sympy
            except ImportError as e:
                warnings.warn(e.msg)
                warnings.warn("please run `pip install sympy` for this functionality.")
                return jacobian_expression
            if info.data["features"] is not None:
                if isinstance(info.data["expression"], str):
                    return (
                        "["
                        + ", ".join(
                            [
                                str(
                                    sympy.S(
                                        info.data["expression"]
                                    ).diff(  # ty: ignore[missing-argument]
                                        key
                                    )
                                )
                                for key in info.data["features"]
                            ],
                        )
                        + "]"
                    )

        return jacobian_expression

    @field_validator("hessian_expression")
    @classmethod
    def set_hessian_expression(cls, hessian_expression, info) -> Union[str, Callable]:
        """Auto-compute Hessian using SymPy for string expressions if not provided.
        Args:            hessian_expression: User-provided Hessian or None.
            info: Pydantic validation context.
        Returns:
            Union[str, Callable]: Hessian expression.
        """
        if (
            hessian_expression is None
            and "features" in info.data.keys()
            and "expression" in info.data.keys()
        ):
            try:
                import sympy
            except ImportError as e:
                warnings.warn(e.msg)
                warnings.warn("please run `pip install sympy` for this functionality.")
                return hessian_expression
            if info.data["features"] is not None:
                if isinstance(info.data["expression"], str):
                    return (
                        "["
                        + ", ".join(
                            [
                                "["
                                + ", ".join(
                                    [
                                        str(
                                            sympy.S(
                                                info.data["expression"]
                                            )  # ty: ignore[missing-argument]
                                            .diff(key1)
                                            .diff(key2)
                                        )
                                        for key1 in info.data["features"]
                                    ]
                                )
                                + "]"
                                for key2 in info.data["features"]
                            ]
                        )
                        + "]"
                    )

        return hessian_expression

    def __call__(
        self, experiments: Union[pd.DataFrame, "_torch.Tensor"]
    ) -> Union[pd.Series, "_torch.Tensor"]:
        """Evaluate the constraint.

        Args:
            experiments: Either a DataFrame with feature columns or a PyTorch tensor

        Returns:
            Constraint values as Series (for DataFrame) or Tensor (for Tensor input)
        """
        # Handle Tensor input from BoTorch
        if _TORCH_AVAILABLE and isinstance(experiments, torch.Tensor):
            # Handle 3D tensor from BoTorch: [n_restarts, q, n_features]
            if experiments.ndim == 3:
                batch_size, q, n_features = experiments.shape
                # Reshape to 2D: [batch_size * q, n_features]
                experiments_2d = experiments.reshape(-1, n_features)
                # Evaluate and reshape back
                results_2d = self.__call__(experiments_2d)
                return results_2d.reshape(batch_size, q)

            if isinstance(self.expression, str):
                # For string expressions, convert tensor to dict
                if experiments.ndim == 1:
                    # Single point: shape (n_features,)
                    feature_dict = {
                        feat: experiments[i] for i, feat in enumerate(self.features)
                    }
                    # Use eval with torch operations available
                    return eval(
                        self.expression,
                        {"__builtins__": {}, "torch": torch},
                        feature_dict,
                    )
                else:
                    # Batch: shape (batch_size, n_features)
                    results = []
                    for point in experiments:
                        feature_dict = {
                            feat: point[i] for i, feat in enumerate(self.features)
                        }
                        result = eval(
                            self.expression,
                            {"__builtins__": {}, "torch": torch},
                            feature_dict,
                        )
                        results.append(result)
                    return torch.stack(results)

            elif isinstance(self.expression, Callable):
                # Callable expression - pass as dict
                if experiments.ndim == 1:
                    feature_dict = {
                        feat: experiments[i] for i, feat in enumerate(self.features)
                    }
                    return self.expression(**feature_dict)
                else:
                    # Batch processing
                    results = []
                    for point in experiments:
                        feature_dict = {
                            feat: point[i] for i, feat in enumerate(self.features)
                        }
                        results.append(self.expression(**feature_dict))
                    return torch.stack(results)

        #  Handle DataFrame input (existing logic)
        if isinstance(self.expression, str):
            return experiments.eval(self.expression, engine="python")
        elif isinstance(self.expression, Callable):
            # Support both:
            # - torch installed: pass torch tensors (enables torch-based callables)
            # - torch not installed: pass numpy arrays (enables numpy-based callables)
            if _TORCH_AVAILABLE:
                func_input = {
                    col: torch.tensor(
                        experiments[col].values,
                        dtype=torch.float64,
                        requires_grad=False,
                    )
                    for col in experiments.columns
                }
                out = self.expression(**func_input)
                if hasattr(out, "detach"):
                    out = out.detach().cpu().numpy()
                return pd.Series(
                    np.asarray(out),
                    index=experiments.index,  # Preserve original indices
                )

            func_input = {
                col: experiments[col].to_numpy() for col in experiments.columns
            }
            out = self.expression(**func_input)
            return pd.Series(np.asarray(out), index=experiments.index)
        raise ValueError("expression must be a string or callable")

    def jacobian(self, experiments: pd.DataFrame) -> pd.DataFrame:
        if self.jacobian_expression is not None:
            if isinstance(self.jacobian_expression, str):
                res = experiments.eval(self.jacobian_expression, engine="python")
                for i, col in enumerate(res):
                    if not hasattr(col, "__iter__"):
                        res[i] = pd.Series(np.repeat(col, experiments.shape[0]))

                if self.features is not None:
                    return pd.DataFrame(
                        res,
                        index=["dg/d" + name for name in self.features],
                    ).transpose()
                return pd.DataFrame(
                    res,
                    index=[f"dg/dx{i}" for i in range(experiments.shape[1])],
                ).transpose()
            elif isinstance(self.jacobian_expression, Callable):
                args = inspect.getfullargspec(self.jacobian_expression).args

                func_input = {
                    arg: torch_tensor(experiments[arg], requires_grad=False)
                    for arg in args
                }
                result = self.jacobian_expression(**func_input)

                return pd.DataFrame(
                    np.array(
                        [
                            result[args.index(col)]
                            if col in args
                            else np.zeros(shape=(len(experiments)))
                            for col in experiments.columns
                        ]
                    ),
                    index=["dg/d" + name for name in experiments.columns],
                ).transpose()
        elif isinstance(self.expression, Callable):
            args = inspect.getfullargspec(self.expression).args

            func_input = tuple(
                [torch_tensor(experiments[arg], requires_grad=False) for arg in args]
            )

            result = torch_jacobian(self.expression, func_input)
            result = [torch_diag(result[i]).cpu().numpy() for i in range(len(args))]

            return pd.DataFrame(
                np.array([result[args.index(col)] for col in args]),
                index=["dg/d" + name for name in args],
            ).transpose()

        raise ValueError(
            "The jacobian of a nonlinear constraint cannot be evaluated if jacobian_expression is None and expression is not Callable.",
        )

    def hessian(self, experiments: pd.DataFrame) -> Dict[Union[str, int], pd.DataFrame]:
        """
        Computes a dict of dataframes where the key dimension is the index of the experiments dataframe
        and the value is the hessian matrix of the constraint evaluated at the corresponding experiment.

        Args:
            experiments (pd.DataFrame): Dataframe to evaluate the constraint Hessian on.

        Returns:
            Dict[pd.DataFrame]: Dictionary of dataframes where the key is the index of the experiments dataframe
            and the value is the Hessian matrix of the constraint evaluated at the corresponding experiment.
        """
        if self.hessian_expression is not None:
            if isinstance(self.hessian_expression, str):
                res = experiments.eval(self.hessian_expression, engine="python")
            else:
                if not isinstance(self.hessian_expression, Callable):
                    raise ValueError(
                        "The hessian_expression must be a string or a callable.",
                    )
                args = inspect.getfullargspec(self.hessian_expression).args

                func_input = {
                    arg: torch_tensor(experiments[arg], requires_grad=False)
                    for arg in args
                }
                res = self.hessian_expression(**func_input)
            for i, _ in enumerate(res):
                for j, entry in enumerate(res[i]):
                    if not hasattr(entry, "__iter__"):
                        res[i][j] = pd.Series(np.repeat(entry, experiments.shape[0]))
            res = np.array(res)
            names = self.features or [f"x{i}" for i in range(experiments.shape[1])]

            return {
                idx: pd.DataFrame(
                    res[..., i],
                    columns=[f"d/d{name}" for name in names],
                    index=[f"d/d{name}" for name in names],
                )
                for i, idx in enumerate(experiments.index)
            }

        elif isinstance(self.expression, Callable):
            args = inspect.getfullargspec(self.expression).args

            func_input = {
                idx: tuple(
                    [
                        torch_tensor(experiments[arg][idx], requires_grad=False)
                        for arg in args
                    ]
                )
                for idx in experiments.index
            }

            names = self.features or [f"x{i}" for i in range(experiments.shape[1])]
            res = {
                idx: pd.DataFrame(
                    np.array(torch_hessian(self.expression, func_input[idx])),
                    columns=[f"d/d{name}" for name in names],
                    index=[f"d/d{name}" for name in names],
                )
                for idx in experiments.index
            }
            return res

        raise ValueError(
            "The hessian of a nonlinear constraint cannot be evaluated if hessian_expression is None and expression is not Callable.",
        )


class NonlinearEqualityConstraint(NonlinearConstraint, EqualityConstraint):
    """Nonlinear equality constraint of the form 'expression == 0'.

    Attributes:
        expression: Mathematical expression that can be evaluated by `pandas.eval`.

    """

    type: Literal["NonlinearEqualityConstraint"] = "NonlinearEqualityConstraint"

    def is_fulfilled(self, experiments: pd.DataFrame, tol: float = 1e-6) -> pd.Series:
        """
        Check if the nonlinear equality constraint is fulfilled.

        Since this constraint is converted to two inequality constraints during
        optimization (f(x) <= tol and f(x) >= -tol), we validate consistently
        by checking if the violation is within the tolerance band.

        Args:
            experiments: DataFrame containing the candidate points to validate
            tol: Tolerance for constraint fulfillment (default: 1e-6)

        Returns:
            Boolean Series indicating whether each candidate fulfills the constraint
        """

        violation = self(experiments)
        # Small epsilon to handle floating-point boundary cases
        # e.g. violation = -0.001 with tol = 0.001 should pass
        # Add a small absolute epsilon to avoid false negatives when we're right on
        # the boundary (e.g. 0.0010000000000001 with tol=0.001).
        eps = max(tol * 1e-9, 1e-15, 1e-9)
        result = pd.Series(np.abs(violation) <= (tol + eps), index=experiments.index)
        return result


class NonlinearInequalityConstraint(NonlinearConstraint, InequalityConstraint):
    """Nonlinear inequality constraint of the form 'expression <= 0'.

    Attributes:
        expression: Mathematical expression that can be evaluated by `pandas.eval`.

    """

    type: Literal["NonlinearInequalityConstraint"] = "NonlinearInequalityConstraint"
