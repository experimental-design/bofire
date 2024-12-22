import inspect
import warnings
from typing import Callable, Literal, Optional, Union

import numpy as np
import pandas as pd
import torch
from pydantic import Field, field_validator
from torch.autograd.functional import jacobian as torch_jacobian

from bofire.data_models.constraints.constraint import (
    EqualityConstraint,
    InequalityConstraint,
    IntrapointConstraint,
)
from bofire.data_models.domain.features import Inputs
from bofire.data_models.features.api import ContinuousInput
from bofire.data_models.types import FeatureKeys


class NonlinearConstraint(IntrapointConstraint):
    """Base class for nonlinear equality and inequality constraints.

    Attributes:
        expression (str): Mathematical expression that can be evaluated by `pandas.eval`.
        jacobian_expression (str): Mathematical expression that that can be evaluated by `pandas.eval`.
        features (list): list of feature keys (str) on which the constraint works on.

    """

    expression: Union[str, Callable]
    features: Optional[FeatureKeys] = Field(default=None, validate_default=True)
    jacobian_expression: Optional[Union[str, Callable]] = Field(
        default=None, validate_default=True
    )

    def validate_inputs(self, inputs: Inputs):
        if self.features is not None:
            keys = inputs.get_keys(ContinuousInput)
            for f in self.features:
                if f not in keys:
                    raise ValueError(
                        f"Feature {f} is not a continuous input feature in the provided Inputs object.",
                    )

    @field_validator("features")
    @classmethod
    def set_features(cls, features, info) -> Optional[FeatureKeys]:
        if "expression" in info.data.keys():
            if isinstance(info.data["expression"], Callable):
                if features is None:
                    return list(inspect.getfullargspec(info.data["expression"]).args)
                else:
                    raise ValueError(
                        "Features must be None if expression is a callable. They will be inferred from the callable.",
                    )

        return features

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

    def __call__(self, experiments: pd.DataFrame) -> pd.Series:
        if isinstance(self.expression, str):
            return experiments.eval(self.expression)
        elif isinstance(self.expression, Callable):
            func_input = {
                col: torch.tensor(experiments[col]) for col in experiments.columns
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

                func_input = {arg: torch.tensor(experiments[arg]) for arg in args}
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
                    index=["dg/d" + name for name in args],
                ).transpose()
        elif isinstance(self.expression, Callable):
            args = inspect.getfullargspec(self.expression).args

            func_input = tuple([torch.tensor(experiments[arg]) for arg in args])

            result = torch_jacobian(self.expression, func_input)
            result = [torch.diag(result[i]).cpu().numpy() for i in range(len(args))]

            return pd.DataFrame(
                np.array([result[args.index(col)] for col in args]),
                index=["dg/d" + name for name in args],
            ).transpose()

        raise ValueError(
            "The jacobian of a nonlinear constraint cannot be evaluated if jacobian_expression is None and expression is not Callable.",
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
