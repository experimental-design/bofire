import warnings
from typing import Literal, Optional

import numpy as np
import pandas as pd
from pydantic import Field, field_validator

from bofire.data_models.constraints.constraint import (
    EqalityConstraint,
    InequalityConstraint,
    IntrapointConstraint,
)
from bofire.data_models.types import FeatureKeys


class NonlinearConstraint(IntrapointConstraint):
    """Base class for nonlinear equality and inequality constraints.

    Attributes:
        expression (str): Mathematical expression that can be evaluated by `pandas.eval`.
        jacobian_expression (str): Mathematical expression that that can be evaluated by `pandas.eval`.
        features (list): list of feature keys (str) on which the constraint works on.
    """

    expression: str
    features: Optional[FeatureKeys] = None
    jacobian_expression: Optional[str] = Field(default=None, validate_default=True)

    @field_validator("jacobian_expression")
    @classmethod
    def set_jacobian_expression(cls, jacobian_expression, info):
        try:
            import sympy  # type: ignore
        except ImportError as e:
            warnings.warn(e.msg)
            warnings.warn("please run `pip install sympy` for this functionality.")
            return jacobian_expression

        if (
            jacobian_expression is None
            and "features" in info.data.keys()
            and "expression" in info.data.keys()
        ):
            if info.data["features"] is not None:
                return (
                    "["
                    + ", ".join(
                        [
                            str(sympy.S(info.data["expression"]).diff(key))
                            for key in info.data["features"]
                        ]
                    )
                    + "]"
                )
        return jacobian_expression

    def __call__(self, experiments: pd.DataFrame) -> pd.Series:
        return experiments.eval(self.expression)

    def jacobian(self, experiments: pd.DataFrame) -> pd.DataFrame:
        if self.jacobian_expression is not None:
            res = experiments.eval(self.jacobian_expression)
            for i, col in enumerate(res):
                if not hasattr(col, "__iter__"):
                    res[i] = pd.Series(np.repeat(col, experiments.shape[0]))

            if self.features is not None:
                return pd.DataFrame(
                    res, index=["dg/d" + name for name in self.features]
                ).transpose()
            else:
                return pd.DataFrame(
                    res, index=[f"dg/dx{i}" for i in range(experiments.shape[1])]
                ).transpose()

        raise ValueError(
            "The jacobian of a nonlinear constraint cannot be evaluated if jacobian_expression is None."
        )


class NonlinearEqualityConstraint(NonlinearConstraint, EqalityConstraint):
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
