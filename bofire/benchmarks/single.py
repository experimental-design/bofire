import math
from typing import Optional

import numpy as np
import pandas as pd
import torch
from botorch.test_functions import Hartmann as botorch_hartmann
from botorch.test_functions.synthetic import Branin as torchBranin
from pydantic.types import PositiveInt
from scipy.stats import dirichlet, multivariate_normal, random_correlation

from bofire.benchmarks.benchmark import Benchmark
from bofire.data_models.constraints.api import NChooseKConstraint
from bofire.data_models.domain.api import Constraints, Domain, Inputs, Outputs
from bofire.data_models.features.api import (
    CategoricalDescriptorInput,
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
    DiscreteInput,
    TaskInput,
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
            **kwargs: Additional arguments for the Benchmark class.

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
                ),
            )

        if self.descriptor:
            input_feature_list.append(
                CategoricalDescriptorInput(
                    key="descriptor",
                    categories=[str(x) for x in range(self.num_categories)],
                    descriptors=["d1"],
                    values=[[x * 2] for x in range(self.num_categories)],
                ),
            )

        # continuous input features
        for d in range(self.dim):
            input_feature_list.append(
                ContinuousInput(key=f"x_{d+1}", bounds=[self.lower, self.upper]),
            )

        # Objective
        output_feature = ContinuousOutput(key="y", objective=MaximizeObjective(w=1))

        self._domain = Domain(
            inputs=Inputs(features=input_feature_list),
            outputs=Outputs(features=[output_feature]),
        )

    def _f(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:  # type: ignore
        """Evaluates benchmark function.

        Args:
            X (pd.DataFrame): Input values. Columns are x_1 and x_2
            **kwargs: Allow additional unused arguments to prevent errors.

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
            pd.DataFrame: x values of optima. Columns are x_1, x_2, y and valid_y

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
                    ContinuousInput(key=f"x_{i}", bounds=[0, 1]) for i in range(dim)
                ],
            ),
            outputs=Outputs(
                features=[ContinuousOutput(key="y", objective=MinimizeObjective())],
            ),
            constraints=(
                Constraints(
                    constraints=[
                        NChooseKConstraint(
                            features=[f"x_{i}" for i in range(dim)],
                            min_count=0,
                            max_count=allowed_k,
                            none_also_valid=True,
                        ),
                    ],
                )
                if allowed_k
                else Constraints()
            ),
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
                        candidates[[f"x_{i}" for i in range(self.dim)]].values,
                    ),
                ),
                "valid_y": [1 for _ in range(len(candidates))],
            },
        )


class Hartmann6plus(Benchmark):
    def __init__(self, dim: int = 6, allowed_k: Optional[int] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._domain = Domain(
            inputs=Inputs(
                features=[
                    ContinuousInput(key=f"x_{i}", bounds=[0, 1]) for i in range(dim)
                ]
            ),
            outputs=Outputs(
                features=[ContinuousOutput(key="y", objective=MinimizeObjective())]
            ),
            constraints=(
                Constraints(
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
                else Constraints()
            ),
        )
        if dim < 6:
            raise ValueError("Hartmann6plus available for dim>=6.")
        self._hartmann = botorch_hartmann(dim=6)

    def get_optima(self) -> pd.DataFrame:
        if len(self.domain.constraints) > 0:
            raise ValueError("Not defined for NChooseK use case.")
        return pd.DataFrame(
            columns=[f"x_{i}" for i in range(6)] + ["y"],
            data=[[0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573, -3.32237]],
        )

    @property
    def dim(self) -> int:
        return len(self.domain.inputs)

    def _f(self, candidates: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "y": self._hartmann(
                    torch.from_numpy(candidates[[f"x_{i}" for i in range(6)]].values)
                ),
                "valid_y": [1 for _ in range(len(candidates))],
            }
        )


class Branin(Benchmark):
    def __init__(self, locality_factor: Optional[float] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._domain = Domain(
            inputs=Inputs(
                features=[
                    ContinuousInput(
                        key="x_1",
                        bounds=[-5.0, 10],
                        local_relative_bounds=[
                            0.5 * locality_factor,
                            0.5 * locality_factor,
                        ]
                        if locality_factor is not None
                        else None,
                    ),
                    ContinuousInput(
                        key="x_2",
                        bounds=[0.0, 15.0],
                        local_relative_bounds=[
                            1.5 * locality_factor,
                            1.5 * locality_factor,
                        ]
                        if locality_factor is not None
                        else None,
                    ),
                ],
            ),
            outputs=Outputs(
                features=[ContinuousOutput(key="y", objective=MinimizeObjective())],
            ),
        )
        self.branin = torchBranin().to(**tkwargs)

    def _f(self, candidates: pd.DataFrame) -> pd.DataFrame:
        c = torch.from_numpy(candidates[self.domain.inputs.get_keys()].values).to(
            **tkwargs,
        )
        return pd.DataFrame(
            {
                "y": self.branin(c).detach().numpy(),
                "valid_y": np.ones(len(candidates)),
            },
        )

    def get_optima(self) -> pd.DataFrame:
        return pd.DataFrame(
            data=np.array(
                [
                    [-math.pi, 12.275, 0.397887],
                    [math.pi, 2.275, 0.397887],
                    [9.42478, 2.475, 0.397887],
                ],
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
                    ContinuousInput(key=f"x_{i+1:02d}", bounds=[0, 1])
                    for i in range(30)
                ],
            ),
            outputs=Outputs(
                features=[ContinuousOutput(key="y", objective=MinimizeObjective())],
            ),
        )
        self.branin = torchBranin().to(**tkwargs)

    def _f(self, candidates: pd.DataFrame) -> pd.DataFrame:
        lb, ub = self.branin.bounds
        c = torch.from_numpy(candidates[self.domain.inputs.get_keys()].values).to(
            **tkwargs,
        )
        return pd.DataFrame(
            {
                "y": self.branin(lb + (ub - lb) * c[..., :2]).detach().numpy(),
                "valid_y": np.ones(len(candidates)),
            },
        )


class Himmelblau(Benchmark):
    """Himmelblau function for testing optimization algorithms
    Link to the definition: https://en.wikipedia.org/wiki/Himmelblau%27s_function
    """

    def __init__(self, use_constraints: bool = False, **kwargs):
        """Initialiszes class of type Himmelblau.

        Args:
            use_constraints (bool, optional): Whether constraints should be used or not (Not implemented yet.). Defaults to False.
            **kwargs: Additional arguments for the Benchmark class.

        Raises:
            ValueError: As constraints are not implemented yet, a True value for use_constraints yields a ValueError.

        """
        super().__init__(**kwargs)
        self.use_constraints = use_constraints
        inputs = []

        inputs.append(ContinuousInput(key="x_1", bounds=[-6, 6]))
        inputs.append(ContinuousInput(key="x_2", bounds=[-6, 6]))

        objective = MinimizeObjective(w=1.0)
        output_feature = ContinuousOutput(key="y", objective=objective)
        if self.use_constraints:
            raise ValueError("Not implemented yet!")
        self._domain = Domain(
            inputs=Inputs(features=inputs),
            outputs=Outputs(features=[output_feature]),
        )

    def _f(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:  # type: ignore
        """Evaluates benchmark function.

        Args:
            X (pd.DataFrame): Input values. Columns are x_1 and x_2
            **kwargs: Allow additional unused arguments to prevent errors.

        Returns:
            pd.DataFrame: y values of the function. Columns are y and valid_y.

        """
        X_temp = X.eval(
            "y=((x_1**2 + x_2 - 11)**2+(x_1 + x_2**2 -7)**2)",
            inplace=False,
        )
        Y = pd.DataFrame({"y": X_temp["y"], "valid_y": 1})
        return Y

    def get_optima(self) -> pd.DataFrame:
        """Returns positions of optima of the benchmark function.

        Returns:
            pd.DataFrame: x values of optima. Columns are x_1 and x_2

        """
        x = np.array(
            [
                [3.0, 2.0],
                [-2.805118, 3.131312],
                [-3.779310, -3.283186],
                [3.584428, -1.848126],
            ],
        )
        y = np.zeros(4)
        return pd.DataFrame(
            np.c_[x, y],
            columns=self.domain.inputs.get_keys() + self.domain.outputs.get_keys(),
        )


class MultiTaskHimmelblau(Benchmark):
    """Himmelblau function for testing optimization algorithms
    Link to the definition: https://en.wikipedia.org/wiki/Himmelblau%27s_function
    """

    def __init__(self, use_constraints: bool = False, **kwargs):
        """Initialiszes class of type Himmelblau.

        Args:
            best_possible_f (float, optional): Not implemented yet. Defaults to 0.0.
            use_constraints (bool, optional): Whether constraints should be used or not (Not implemented yet.). Defaults to False.
            **kwargs: Additional arguments for the Benchmark class.

        Raises:
            ValueError: As constraints are not implemented yet, a True value for use_constraints yields a ValueError.

        """
        super().__init__(**kwargs)
        self.use_constraints = use_constraints
        inputs = []

        inputs.append(TaskInput(key="task_id", categories=["task_1", "task_2"]))
        inputs.append(ContinuousInput(key="x_1", bounds=[-6, 6]))
        inputs.append(ContinuousInput(key="x_2", bounds=[-6, 6]))

        objective = MinimizeObjective(w=1.0)
        output_feature = ContinuousOutput(key="y", objective=objective)
        if self.use_constraints:
            raise ValueError("Not implemented yet!")
        self._domain = Domain(
            inputs=Inputs(features=inputs),
            outputs=Outputs(features=[output_feature]),
        )

    def _f(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:  # type: ignore
        """Evaluates benchmark function.

        Args:
            X (pd.DataFrame): Input values. Columns are x_1 and x_2
            **kwargs: Allow additional unused arguments to prevent errors.

        Returns:
            pd.DataFrame: y values of the function. Columns are y and valid_y.

        """
        # initialize y outputs
        Y = pd.DataFrame({"y": np.zeros(len(X)), "valid_y": 0})
        # evaluate task 1
        X_temp = X.query("task_id == 'task_1'").eval(
            "y=((x_1**2 + x_2 - 11)**2+(x_1 + x_2**2 -7)**2)",
            inplace=False,
        )
        Y.loc[X_temp.index, "y"] = X_temp["y"]
        Y.loc[X_temp.index, "valid_y"] = 1
        # evaluate task 2
        X_temp = X.query("task_id == 'task_2'").eval(
            "y=((x_1**2 + x_2 - 11)**2+(x_1 + x_2**2 -7)**2) + x_1 * x_2",
            inplace=False,
        )
        Y.loc[X_temp.index, "y"] = X_temp["y"]
        Y.loc[X_temp.index, "valid_y"] = 1
        return Y

    def get_optima(self) -> pd.DataFrame:
        """Returns positions of optima of the benchmark function.

        Returns:
            pd.DataFrame: x values of optima. Columns are x_1, x_2, task_id

        """
        out = [
            [3.0, 2.0, "task_1", 0],
            [-2.805118, 3.131312, "task_1", 0],
            [-3.779310, -3.283186, "task_1", 0],
            [3.584428, -1.848126, "task_1", 0],
        ]

        return pd.DataFrame(
            out,
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
        raise NotImplementedError


class Multinormalpdfs(Benchmark):
    """The sum of probability density functions of multivariate Gaussian distributions

    A virtual experiment in this benchmark class is evaluated by summing one or more
    multivariate normal PDFs:
        y_expt = sum_i {const_i * exp(-0.5 * (x - mu_i)^T Sigma_i^-1 (x - mu_i))}
    where mu_i is the i-th mean vector and Sigma_i is the i-th covariance matrix.

    This allows us to control
        * the possible presence of multiple optima by choosing the number of Gaussians
        * the position of the optimum/optima by setting the mean(s)
        * the axis-alignment of each elliptical dip in the objective landscape by modifying the
          covariance matrix/matrices
        * the presence of non-influential variables by setting the corresponding elements in the
          covariance matrix to zero or a large number
    """

    def __init__(
        self,
        dim: int = 5,
        n_gaussians: int = 1,
        stdev: float = 0.4,
        eigscale: float = 0.5,
        opt_on_boundary: bool = False,
        N_unimportant_inputs: int = 2,
        seed: Optional[int] = None,
        means: Optional[list] = None,
        covmats: Optional[list] = None,
        **kwargs,
    ) -> None:
        """Initializes the class of type Multinormalpdfs

        Args:
            dim : number of input dimensions
            n_gaussians : number of gaussian pdfs in the sum
            stdev : standard deviation used to generate the covariance matrices
            eigscale : the concentration parameter (this value repeated dim times)
                of a Dirichlet distribution used to sample scaled eigenvalues of the
                correlation matrix, which is used to create the covariance matrix. Larger values
                will make the covariance matrix more dominated by the diagonal and thus the shape
                of the objective landscape nice and axis-parallel. Smaller values will
                emphasize the off-diagonal elements of the covariance matrix. See details.
            opt_on_boundary : if True, the first element of the mean vector(s) is set to
                zero to put the optimum on the boundary of the space
            N_unimportant_inputs : this many inputs receive zeroed rows and columns in the
                covariance matrix and a large number on the diagonal. This essentially
                makes them noise variables that don't do anything (or only have a very
                weak effect)
            seed (int, optional): random seed. Defaults to None.
            means : a list of mean vectors in case the user wants to specify them and
                bypass generation. Setting this causes above args except dim to be ignored
            covmats : a list of covariance matrices in case the user wants to specify them
                and bypass generation. As with means
            **kwargs: Additional arguments for the Benchmark class.

        Details:
            The way the covariance matrix is generated is perhaps nontrivial:
            1) sample n_dims values from a dirichlet distribution, and call this sample eigs. sum(eigs)=1
            2) scale eigs so that the sum equals n_dims
            3) generate a random correlation matrix with the eigenvalues eigs
            4) make this into a covariance matrix using cov_mat = stdevs @ corrmat @ stdevs
            5) make any required additional changes to make the PDF almost flat in some directions

        """
        super().__init__(**kwargs)
        self.dim = dim
        self.n_gaussians = n_gaussians
        self.stdev = stdev
        self.eigscale = eigscale
        self.opt_on_boundary = opt_on_boundary
        self.N_unimportant_inputs = N_unimportant_inputs
        self._domain = Domain(
            inputs=Inputs(
                features=[
                    ContinuousInput(key=f"x_{i}", bounds=[0, 1])
                    for i in range(self.dim)
                ],
            ),
            outputs=Outputs(
                features=[ContinuousOutput(key="y", objective=MaximizeObjective())],
            ),
        )
        np.random.seed(seed)

        gaussians = []
        prefactors = []
        if means is not None and covmats is not None:
            # user has define the parameters of the distributions
            for mean, cov_mat in zip(means, covmats):
                if len(mean) != dim:
                    raise ValueError(
                        "Length of mean should equal dimensionality in Multinormalpdfs",
                    )
                gaussians.append(multivariate_normal(mean=mean, cov=cov_mat))
            n_gaussians = len(gaussians)
        else:
            # Generate the multivariate normal distributions
            unimportant_dims = np.random.choice(
                list(range(self.dim)),
                self.N_unimportant_inputs,
                replace=False,
            )
            for _ in range(n_gaussians):
                mean = np.random.random(size=dim)
                if opt_on_boundary:
                    mean[0] = 0.0
                eigs = np.ravel(dirichlet.rvs(alpha=[eigscale] * dim, size=1))
                eigs = dim * eigs
                corrmat = random_correlation.rvs(eigs)
                stdevs = np.diag([stdev] * dim)
                cov_mat = stdevs @ corrmat @ stdevs
                for i in unimportant_dims:
                    cov_mat[i, :] = 0.0
                    cov_mat[:, i] = 0.0
                    cov_mat[i, i] = 10.0
                gaussians.append(multivariate_normal(mean=mean, cov=cov_mat))
        for i in range(n_gaussians):
            prefactors.append(
                (2 * np.pi) ** (-dim / 2) / np.sqrt(np.linalg.det(gaussians[i].cov))
            )
        self.gaussians = gaussians
        self.prefactors = prefactors

    def _f(self, X: pd.DataFrame) -> pd.DataFrame:  # type: ignore
        return pd.DataFrame(
            {
                "y": X.apply(
                    lambda x: sum(
                        [
                            g.pdf(x.to_numpy()) / prefac
                            for prefac, g in zip(self.prefactors, self.gaussians)
                        ]
                    ),
                    axis=1,
                ),
                "valid_y": np.ones(len(X)),
            },
        )

    def get_optima(self) -> pd.DataFrame:
        if self.n_gaussians != 1:
            raise NotImplementedError(
                "Position of optima only implemented for benchmark with n_gaussians = 1",
            )
        x_opt = pd.DataFrame(
            {f"x_{i}": self.gaussians[0].mean[i] for i in range(self.dim)},
            index=[0],
        )
        return pd.concat([x_opt, self._f(x_opt)], axis=1)
