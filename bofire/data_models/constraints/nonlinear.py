import inspect
import warnings
from typing import Callable, Dict, Literal, Optional, Union

import numpy as np
import pandas as pd
from pydantic import Field, field_validator, model_validator


try:
    import torch
    from torch.autograd.functional import hessian as torch_hessian
    from torch.autograd.functional import jacobian as torch_jacobian

    torch_tensor = torch.tensor
    torch_diag = torch.diag
except ImportError:

    def error_func(*args, **kwargs):
        raise NotImplementedError("torch must be installed to use this functionality")

    torch_jacobian = error_func
    torch_tensor = error_func
    torch_diag = error_func
    torch_hessian = error_func

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
        keys = inputs.get_keys(ContinuousInput)
        for f in self.features:
            if f not in keys:
                raise ValueError(
                    f"Feature {f} is not a continuous input feature in the provided Inputs object.",
                )

    @model_validator(mode="after")
    def validate_features(self):
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
                                str(sympy.S(info.data["expression"]).diff(key))
                                for key in info.data["features"]
                            ],
                        )
                        + "]"
                    )

        return jacobian_expression

    @field_validator("hessian_expression")
    @classmethod
    def set_hessian_expression(cls, hessian_expression, info) -> Union[str, Callable]:
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
                                            sympy.S(info.data["expression"])
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

    def __call__(self, experiments: pd.DataFrame) -> pd.Series:
        if isinstance(self.expression, str):
            return experiments.eval(self.expression)
        elif isinstance(self.expression, Callable):
            func_input = {
                col: torch_tensor(experiments[col], requires_grad=False)
                for col in experiments.columns
            }
            return pd.Series(self.expression(**func_input).cpu().numpy())

    def jacobian(self, experiments: pd.DataFrame) -> pd.DataFrame:
        if self.jacobian_expression is not None:
            if isinstance(self.jacobian_expression, str):
                res = experiments.eval(self.jacobian_expression)
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
                res = experiments.eval(self.hessian_expression)
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


class NonlinearInequalityConstraint(NonlinearConstraint, InequalityConstraint):
    """Nonlinear inequality constraint of the form 'expression <= 0'.

    Attributes:
        expression: Mathematical expression that can be evaluated by `pandas.eval`.

    """

    type: Literal["NonlinearInequalityConstraint"] = "NonlinearInequalityConstraint"
