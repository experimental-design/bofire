import math
from typing import Optional

import numpy as np
import pandas as pd
import torch
from botorch.test_functions import Hartmann as botorch_hartmann
from botorch.test_functions.synthetic import Branin as torchBranin
from pydantic.types import PositiveInt

from bofire.benchmarks.benchmark import Benchmark
from bofire.data_models.constraints.api import NChooseKConstraint
from bofire.data_models.domain.api import Constraints, Domain, Inputs, Outputs
from bofire.data_models.features.api import (
    CategoricalDescriptorInput,
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
    DiscreteInput,
)
from bofire.data_models.objectives.api import MaximizeObjective, MinimizeObjective
from bofire.utils.torch_tools import tkwargs


class Ackley(Benchmark):
    """Ackley function for testing optimization algorithms
    Virtual experiment corresponds to a function evaluation.
    Examples
    --------
    >>> b = Ackley()
    >>> columns = [v.name for v in b.domain.variables]
    >>> values = [v.bounds[0]+0.1*(v.bounds[1]-v.bounds[0]) for v in b.domain.variables]
    >>> values = np.array(values)
    >>> conditions = DataSet(values, columns=columns)
    >>> results = b.run_experiments(conditions)
    Notes
    -----
    This function is the negated version of https://en.wikipedia.org/wiki/Ackley_function.
    """

    # @validator("validate_categoricals")
    # def validate_categoricals(cls, v, num_categoricals):
    #     if v and num_categoricals ==1:
    #         raise ValueError("num_categories  must be specified if categorical=True")
    #     return v

    def __init__(
        self,
        num_categories: PositiveInt = 3,
        categorical: bool = False,
        descriptor: bool = False,
        dim: PositiveInt = 2,
        lower: float = -32.768,
        upper: float = 32.768,
        best_possible_f: float = 0.0,
        evaluated_points: Optional[list] = None,
        **kwargs,
    ):
        """Initializes benchmark function of type Ackley.

        Args:
            num_categories (PositiveInt, optional): Number of categories. Defaults to 3.
            categorical (bool, optional): Use categorical inputs. Defaults to False.
            descriptor (bool, optional): Use descriptive inputs. Defaults to False.
            dim (PositiveInt, optional): Dimension. Defaults to 2.
            lower (float, optional): Upper boundary. Defaults to -32.768.
            upper (float, optional): Lower boundary. Defaults to 32.768.
            best_possible_f (float, optional): Best possible function value. Defaults to 0.0.
            evaluated_points (list, optional): Evaluated points. Defaults to [].
        """
        super().__init__(**kwargs)
        self.num_categories = num_categories
        self.categorical = categorical
        self.descriptor = descriptor
        self.dim = dim
        self.lower = lower
        self.upper = upper
        self.best_possible_f = best_possible_f
        if evaluated_points is None:
            evaluated_points = []
        self.evaluated_points = evaluated_points

        input_feature_list = []
        # Decision variables
        if self.categorical:
            input_feature_list.append(
                CategoricalInput(
                    key="category",
                    categories=[str(x) for x in range(self.num_categories)],
                )
            )

        if self.descriptor:
            input_feature_list.append(
                CategoricalDescriptorInput(
                    key="descriptor",
                    categories=[str(x) for x in range(self.num_categories)],
                    descriptors=["d1"],
                    values=[[x * 2] for x in range(self.num_categories)],
                )
            )

        # continuous input features
        for d in range(self.dim):
            input_feature_list.append(
                ContinuousInput(key=f"x_{d+1}", bounds=(self.lower, self.upper))
            )

        # Objective
        output_feature = ContinuousOutput(key="y", objective=MaximizeObjective(w=1))

        self._domain = Domain(
            inputs=Inputs(features=input_feature_list),
            outputs=Outputs(features=[output_feature]),
        )

    def _f(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Evaluates benchmark function.

        Args:
            X (pd.DataFrame): Input values. Columns are x_1 and x_2

        Returns:
            pd.DataFrame: y values of the function. Columns are y and valid_y.
        """
        a = 20
        b = 0.2
        c = np.pi * 2
        x = np.array([X[f"x_{d+1}"] for d in range(self.dim)])

        c = np.zeros(len(X))
        d = np.zeros(len(X))
        n = self.dim

        if self.categorical:
            # c = pd.to_numeric(X["category"], downcast="float")
            c = X.loc[:, "category"].values.astype(np.float64)
        if self.descriptor:
            d = X.loc[:, "descriptor"].values.astype(np.float64)

        z = x + c + d

        term1 = -a * np.exp(-b * ((1 / n) * np.sum(z**2, axis=0)) ** 0.5)
        term2 = -np.exp((1 / n) * np.sum(np.cos(c * z), axis=0))
        term3 = a + np.exp(1)
        y = term1 + term2 + term3
        Y = pd.DataFrame({"y": y, "valid_y": 1})
        # save evaluated points for plotting
        self.evaluated_points.append(x.tolist())
        return Y

    def get_optima(self) -> pd.DataFrame:
        """Returns positions of optima of the benchmark function.

        Returns:
            pd.DataFrame: x values of optima. Colums are x_1, x_2, y and valid_y
        """
        x = np.zeros((1, self.dim))
        y = 0
        return pd.DataFrame(
            np.c_[x, y],
            columns=self.domain.inputs.get_keys() + self.domain.outputs.get_keys(),
        )


class Hartmann(Benchmark):
    def __init__(self, dim: int = 6, allowed_k: Optional[int] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._domain = Domain(
            inputs=Inputs(
                features=[
                    ContinuousInput(key=f"x_{i}", bounds=(0, 1)) for i in range(dim)
                ]
            ),
            outputs=Outputs(
                features=[ContinuousOutput(key="y", objective=MinimizeObjective())]
            ),
            constraints=Constraints(
                constraints=[
                    NChooseKConstraint(
                        features=[f"x_{i}" for i in range(dim)],
                        min_count=0,
                        max_count=allowed_k,
                        none_also_valid=True,
                    )
                ]
            )
            if allowed_k
            else Constraints(),
        )
        self._hartmann = botorch_hartmann(dim=dim)

    def get_optima(self) -> pd.DataFrame:
        if self.dim != 6:
            raise ValueError("Only available for dim==6.")
        if len(self.domain.constraints) > 0:
            raise ValueError("Not defined for NChooseK use case.")
        return pd.DataFrame(
            columns=[f"x_{i}" for i in range(self.dim)] + ["y"],
            data=[[0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573, -3.32237]],
        )

    @property
    def dim(self) -> int:
        return len(self.domain.inputs)

    def _f(self, candidates: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "y": self._hartmann(
                    torch.from_numpy(
                        candidates[[f"x_{i}" for i in range(self.dim)]].values
                    )
                ),
                "valid_y": [1 for _ in range(len(candidates))],
            }
        )


class Branin(Benchmark):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._domain = Domain(
            inputs=Inputs(
                features=[
                    ContinuousInput(key="x_1", bounds=(-5.0, 10)),
                    ContinuousInput(key="x_2", bounds=(0.0, 15.0)),
                ]
            ),
            outputs=Outputs(
                features=[ContinuousOutput(key="y", objective=MinimizeObjective())]
            ),
        )
        self.branin = torchBranin().to(**tkwargs)

    def _f(self, candidates: pd.DataFrame) -> pd.DataFrame:
        c = torch.from_numpy(candidates[self.domain.inputs.get_keys()].values).to(
            **tkwargs
        )
        return pd.DataFrame(
            {
                "y": self.branin(c).detach().numpy(),
                "valid_y": np.ones(len(candidates)),
            }
        )

    def get_optima(self) -> pd.DataFrame:
        return pd.DataFrame(
            data=np.array(
                [
                    [-math.pi, 12.275, 0.397887],
                    [math.pi, 2.275, 0.397887],
                    [9.42478, 2.475, 0.397887],
                ]
            ),
            columns=self.domain.inputs.get_keys() + self.domain.outputs.get_keys(),
        )


class Branin30(Benchmark):
    """Thirty dimensional Branin function in which only the first two dimensions are used to
    evaluate the actual Branin. Source: https://github.com/pytorch/botorch/blob/main/tutorials/saasbo.ipynb.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._domain = Domain(
            inputs=Inputs(
                features=[
                    ContinuousInput(key=f"x_{i+1:02d}", bounds=(0, 1))
                    for i in range(30)
                ]
            ),
            outputs=Outputs(
                features=[ContinuousOutput(key="y", objective=MinimizeObjective())]
            ),
        )
        self.branin = torchBranin().to(**tkwargs)

    def _f(self, candidates: pd.DataFrame) -> pd.DataFrame:
        lb, ub = self.branin.bounds  # type: ignore
        c = torch.from_numpy(candidates[self.domain.inputs.get_keys()].values).to(
            **tkwargs
        )
        return pd.DataFrame(
            {
                "y": self.branin(lb + (ub - lb) * c[..., :2]).detach().numpy(),
                "valid_y": np.ones(len(candidates)),
            }
        )


class Himmelblau(Benchmark):
    """Himmelblau function for testing optimization algorithms
    Link to the definition: https://en.wikipedia.org/wiki/Himmelblau%27s_function
    """

    def __init__(self, use_constraints: bool = False, **kwargs):
        """Initialiszes class of type Himmelblau.

        Args:
            best_possible_f (float, optional): Not implemented yet. Defaults to 0.0.
            use_constraints (bool, optional): Whether constraints should be used or not (Not implemented yet.). Defaults to False.

        Raises:
            ValueError: As constraints are not implemeted yet, a True value for use_constraints yields a ValueError.
        """
        super().__init__(**kwargs)
        self.use_constraints = use_constraints
        inputs = []

        inputs.append(ContinuousInput(key="x_1", bounds=(-6, 6)))
        inputs.append(ContinuousInput(key="x_2", bounds=(-6, 6)))

        objective = MinimizeObjective(w=1.0)
        output_feature = ContinuousOutput(key="y", objective=objective)
        if self.use_constraints:
            raise ValueError("Not implemented yet!")
        self._domain = Domain(
            inputs=Inputs(features=inputs),
            outputs=Outputs(features=[output_feature]),
        )

    def _f(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Evaluates benchmark function.

        Args:
            X (pd.DataFrame): Input values. Columns are x_1 and x_2

        Returns:
            pd.DataFrame: y values of the function. Columns are y and valid_y.
        """
        X_temp = X.eval(
            "y=((x_1**2 + x_2 - 11)**2+(x_1 + x_2**2 -7)**2)", inplace=False
        )
        Y = pd.DataFrame({"y": X_temp["y"], "valid_y": 1})
        return Y

    def get_optima(self) -> pd.DataFrame:
        """Returns positions of optima of the benchmark function.

        Returns:
            pd.DataFrame: x values of optima. Colums are x_1 and x_2
        """
        x = np.array(
            [
                [3.0, 2.0],
                [-2.805118, 3.131312],
                [-3.779310, -3.283186],
                [3.584428, -1.848126],
            ]
        )
        y = np.zeros(4)
        return pd.DataFrame(
            np.c_[x, y],
            columns=self.domain.inputs.get_keys() + self.domain.outputs.get_keys(),
        )


class DiscreteHimmelblau(Himmelblau):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        inputs = []

        inputs.append(DiscreteInput(key="x_1", values=np.linspace(-6, 6, 20).tolist()))
        inputs.append(DiscreteInput(key="x_2", values=np.linspace(-6, 6, 20).tolist()))

        objective = MinimizeObjective(w=1.0)
        output_feature = ContinuousOutput(key="y", objective=objective)

        self._domain = Domain(
            inputs=Inputs(features=inputs),
            outputs=Outputs(features=[output_feature]),
        )


class _CategoricalDiscreteHimmelblau(Himmelblau):  # only used for testing
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        inputs = []

        inputs.append(DiscreteInput(key="x_1", values=np.linspace(-6, 6, 20).tolist()))
        inputs.append(DiscreteInput(key="x_2", values=np.linspace(-6, 6, 20).tolist()))
        inputs.append(CategoricalInput(key="x_3", categories=["a", "b", "c"]))

        objective = MinimizeObjective(w=1.0)
        output_feature = ContinuousOutput(key="y", objective=objective)

        self._domain = Domain(
            inputs=Inputs(features=inputs),
            outputs=Outputs(features=[output_feature]),
        )

    def get_optima(self):
        raise NotImplementedError()
