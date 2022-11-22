from typing import List, Optional

import numpy as np
import pandas as pd
from pydantic.types import PositiveInt

from bofire.domain import Domain
from bofire.domain.constraints import (
    LinearEqualityConstraint,
    LinearInequalityConstraint,
)
from bofire.domain.features import (
    CategoricalDescriptorInput,
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
    OutputFeature,
)
from bofire.domain.objectives import MaximizeObjective, MinimizeObjective
from bofire.utils.study import Study


class SingleObjective(Study):

    def __init__(self,**data):
        super().__init__(**data)
        if len(self.domain.output_features.get_by_objective(excludes=None)) > 1: #TODO: update, when more features without DesFunc are implemented!
            raise ValueError("received multiobjective domain.")

    #TODO maybe unite with get_fbest from sobo, but not every strategy has get_fbest so far
    # and we have no universal way to compute it in domain --> maybe implement it also there.
    def get_fbest(self, experiments: Optional[pd.DataFrame]=None):
        if experiments is None:
            experiments = self.experiments
        ofeat = self.domain.output_features.get_by_objective(excludes=None)[0] 
        desirability = ofeat.desirability_function(experiments[ofeat.key])
        return experiments.at[desirability.argmax(),ofeat.key]


class Himmelblau(SingleObjective):

    use_constraints: bool = False
    best_possible_f: float = 0.0

    def setup_domain(self):
        domain = Domain()

        domain.add_feature(ContinuousInput(key="x_1",lower_bound=-4., upper_bound=4.))
        domain.add_feature(ContinuousInput(key="x_2",lower_bound=-4., upper_bound=6.)) # ToDo, check for correct bounds

        desirability_function = MinimizeObjective(w=1.)
        domain.add_feature(ContinuousOutput(key="y",desirability_function=desirability_function))

        if self.use_constraints:
            raise ValueError("Not implemented yet!")
        return domain

    def run_candidate_experiments(self,candidates:pd.DataFrame, **kwargs):
        candidates.eval("y=((x_1**2 + x_2 - 11)**2+(x_1 + x_2**2 -7)**2)",inplace=True)
        candidates["valid_y"] =1
        return candidates[self.domain.experiment_column_names].copy()

class Hartmann3D(SingleObjective):
    """Hartmann test function (3D) for testing optimization algorithms
    Virtual experiment corresponds to a function evaluation.
    Examples
    --------
    >>> b = Hartmann3D()
    >>> columns = [v.name for v in b.domain.variables]
    >>> values = [v.bounds[0]+0.1*(v.bounds[1]-v.bounds[0]) for v in b.domain.variables]
    >>> values = np.array(values)
    >>> values = np.atleast_2d(values)
    >>> conditions = DataSet(values, columns=columns)
    >>> results = b.run_experiments(conditions)
    Notes
    -----
    This function is taken from https://www.sfu.ca/~ssurjano/hart3.html.
    """
    use_constraints: bool = False
    minimize: bool = True
    evaluated_points: Optional[List]=[]

    def setup_domain(self):
        domain = Domain()

        # Input features
        feature1 = ContinuousInput(key="x_1", lower_bound=0, upper_bound=1)
        feature2 = ContinuousInput(key="x_2", lower_bound=0, upper_bound=1)
        feature3 = ContinuousInput(key="x_3", lower_bound=0, upper_bound=1)
        
        # target
        output1 = ContinuousOutput(key="y", desirability_function = MaximizeObjective(w=1))
        
        # constraints 
        constraints = []
        if self.use_constraints:
            constraint1 = LinearInequalityConstraint(features=['y'], coefficients=[1], rhs=0) #TODO: check constraint direction!
            constraint2 = LinearInequalityConstraint(features=['y'], coefficients=[-1], rhs=-3.9) #TODO: check constraint direction!
            constraint3 = LinearInequalityConstraint(features=['x_1','x_2','x_3'], coefficients=[1, 1, 1], rhs=1.625)
            constraints = [constraint1, constraint2, constraint3]
        
         
        domain = Domain(input_features=[feature1, feature2, feature3], output_features=[output1], constraints=constraints)
        return domain

    def run_candidate_experiments(self, candidates, **kwargs):
        def function_evaluation(x_1, x_2, x_3):
            x_exp = np.asarray([x_1, x_2, x_3])
            A = np.array([[3, 10, 30], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35]])
            P = (
                np.array(
                    [
                        [3689, 1170, 2673],
                        [4699, 4387, 7470],
                        [1091, 8732, 5547],
                        [381, 5743, 8828],
                    ]
                )
                * 10 ** (-4)
            )
            alpha = np.array([1, 1.2, 3.0, 3.2])
            d = np.zeros((4, 1))
            for k in range(4):
                d[k] = np.sum(np.dot(A[k, :], (x_exp - P[k, :]) ** 2))
            y = np.sum(np.dot(alpha, np.exp(-d))).T
            if self.minimize:
                y = -y
            return y

        x_1 = float(candidates["x_1"])
        x_2 = float(candidates["x_2"])
        x_3 = float(candidates["x_3"])

        y = function_evaluation(x_1, x_2, x_3)
        candidates["y"] = y
        candidates["valid_y"] = 1

        # save evaluated points for plotting
        self.evaluated_points.append([x_1, x_2, x_3, y])

        return candidates[self.domain.experiment_column_names].copy()

    def plot(self, ax=None, **kwargs):
        """Make a plot of the experiments evaluated thus far
        Parameters
        ----------
        ax: `matplotlib.pyplot.axes`, optional
            An existing axis to apply the plot to
        Returns
        -------
        if ax is None returns a tuple with the first component
        as the a new figure and the second component the axis
        if ax is a matplotlib axis, returns only the axis
        Raises
        ------
        ValueError
            If there are no points to plot
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        # evaluated points in <run_experiments> and optional points added by the user
        extra_points = kwargs.get("extra_points", None)

        # polygons objects to be plotted
        polygons = kwargs.get("polygons", None)

        points = self.evaluated_points
        if extra_points is None:
            if points is None:
                raise ValueError("No points to plot.")
        else:
            if points is None:
                points = extra_points
            else:
                points.append(extra_points)

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            return_fig = True
        else:
            return_fig = False

        # get domain bounds and plot frame/axes
        bounds_x_1 = [self.domain.get_feature("x_1").lower_bound, self.domain.get_feature("x_1").lower_bound]
        bounds_x_2 = [self.domain.get_feature("x_2").lower_bound, self.domain.get_feature("x_1").lower_bound]
        bounds_x_3 = [self.domain.get_feature("x_3").lower_bound, self.domain.get_feature("x_1").lower_bound]

        expand_bounds = 0
        ax.axes.set_xlim3d(
            left=bounds_x_1[0] - expand_bounds, right=bounds_x_1[1] + expand_bounds
        )
        ax.axes.set_ylim3d(
            bottom=bounds_x_2[0] - expand_bounds, top=bounds_x_2[1] + expand_bounds
        )
        ax.axes.set_zlim3d(
            bottom=bounds_x_3[0] - expand_bounds, top=bounds_x_3[1] + expand_bounds
        )

        # plot evaluated and extra points with enumeration
        for c in range(len(points)):
            tmp_x_1, tmp_x_2, tmp_x_3, col = (
                points[c][0],
                points[c][1],
                points[c][2],
                points[c][3],
            )
            if self.maximize:
                col = -col
            p = ax.scatter(
                tmp_x_1, tmp_x_2, tmp_x_3, c=col, cmap="seismic", vmin=-4, vmax=0
            )
            ax.text(tmp_x_1 + 0.05, tmp_x_2 + 0.05, tmp_x_3 + 0.05, c + 1, fontsize=7)
        fig.colorbar(p, shrink=0.5)

        # plot constraints
        if len(self.domain.constraints) > 0:
            x = np.linspace(bounds_x_1[0], bounds_x_1[1], 100)
            y = np.linspace(bounds_x_2[0], bounds_x_2[1], 100)
            x_1, x_2 = np.meshgrid(x, y)
            # workaround to plot constraint plane
            x_3 = 0
            for c in self.domain.constraints:
                z = -eval(c.lhs)
                z[z < bounds_x_3[0]] = np.nan
                z[z > bounds_x_3[1]] = np.nan
                ax.plot_surface(
                    x_1,
                    x_2,
                    z,
                    vmin=bounds_x_3[0],
                    vmax=bounds_x_3[1],
                    rstride=4,
                    cstride=4,
                    alpha=0.8,
                )

        # plot polygons
        if polygons:
            for i in range(len(polygons)):
                polygon = Poly3DCollection(polygons[i], alpha=0.1)
                polygon.set_edgecolor("b")
                ax.add_collection3d(polygon)

        if return_fig:
            return fig, ax
        else:
            return ax


class Ackley(SingleObjective):
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
    num_categories: PositiveInt = 3
    categorical: bool = False
    descriptor: bool = False
    dim: PositiveInt = 2
    lower: float = -1
    upper: float = 3
    best_possible_f: float = 0.0
    evaluated_points = []

    # @validator("validate_categoricals")
    # def validate_categoricals(cls, v, num_categoricals):
    #     if v and num_categoricals ==1:
    #         raise ValueError("num_categories  must be specified if categorical=True")
    #     return v

    def setup_domain(self):
        domain = Domain()

        # Decision variables
        if self.categorical:
            domain.add_feature(CategoricalInput(key='category', categories=[str(x) for x in range(self.num_categories)]))
        
        if self.descriptor:
            domain.add_feature(CategoricalDescriptorInput(key='descriptor', categories=[str(x) for x in range(self.num_categories)], descriptors = ['d1'], values=[ [x*2] for x in range(self.num_categories)]))

        # continuous input features
        for d in range(self.dim):
            domain.add_feature(ContinuousInput(key=f"x_{d+1}", lower_bound = self.lower, upper_bound=self.upper))

        # Objective
        domain.add_feature(ContinuousOutput(key="y", desirability_function = MaximizeObjective(w=1)))
        return domain

    def run_candidate_experiments(self, candidates, **kwargs):
        x = np.array([candidates[f"x_{d+1}"] for d in range(self.dim)])
        c = np.zeros(len(candidates))
        d = np.zeros(len(candidates))

        if self.categorical:
            #c = pd.to_numeric(candidates["category"], downcast="float")
            c = candidates.loc[:,"category"].values.astype(np.float)
        if self.descriptor:
            d = candidates.loc[:,"descriptor"].values.astype(np.float) 
        
        z = x + c + d

        first_term = -20 * np.exp(-0.2 * np.sqrt(1 / self.dim * (z ** 2).sum()))
        second_term = -np.exp(1 / self.dim * (np.cos(2 * np.pi * z)).sum())
        y = -(first_term + second_term + 20 + np.exp(1) + (c+d) / 2 )
        
        candidates["y"] = y
        candidates["valid_y"] = 1

        # save evaluated points for plotting
        self.evaluated_points.append(x.tolist())

        return candidates[self.domain.experiment_column_names].copy()

    def plot(self, ax=None, **kwargs):
        """Make a plot of the experiments evaluated thus far
        Parameters
        ----------
        ax: `matplotlib.pyplot.axes`, optional
            An existing axis to apply the plot to
        Returns
        -------
        if ax is None returns a tuple with the first component
        as the a new figure and the second component the axis
        if ax is a matplotlib axis, returns only the axis
        Raises
        ------
        ValueError
            If there are no points to plot
        """
        assert self.dim <= 2, "plot only works for dim <= 2"
        if self.categorical:
            assert (
                self.num_categories == 1
            ), "plot only works for num_categories == 1 if categorical=True"

        import matplotlib.pyplot as plt
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import Polygon

        # evaluated points in <run_experiments> and optional points added by the user
        extra_points = kwargs.get("extra_points", None)

        # polygons objects to be plotted
        polygons = kwargs.get("polygons", None)

        points = self.evaluated_points
        if extra_points is not None:
            points.append(extra_points)
        if extra_points is None:
            if points is None:
                raise ValueError("No points to plot.")
        else:
            if points is None:
                points = extra_points
            else:
                points.append(extra_points)

        if ax is None:
            fig, ax = plt.subplots(1)
            return_fig = True
        else:
            return_fig = False

        # get domain bounds and plot frame/axes
        bounds_x_1 = [self.domain.get_feature("x_1").lower_bound, self.domain.get_feature("x_1").upper_bound]
        bounds_x_2 = [self.domain.get_feature("x_2").lower_bound, self.domain.get_feature("x_2").upper_bound]
        expand_bounds = 1
        plt.axis(
            [
                bounds_x_1[0] - expand_bounds,
                bounds_x_1[1] + expand_bounds,
                bounds_x_2[0] - expand_bounds,
                bounds_x_2[1] + expand_bounds,
            ]
        )

        ax.axvline(x=bounds_x_1[0], color="k", linestyle="--")
        ax.axhline(y=bounds_x_2[0], color="k", linestyle="--")
        ax.axvline(x=bounds_x_1[1], color="k", linestyle="--")
        ax.axhline(y=bounds_x_2[1], color="k", linestyle="--")

        # plot contour
        xlist = np.linspace(
            bounds_x_1[0] - expand_bounds, bounds_x_1[1] + expand_bounds, 1000
        )
        ylist = np.linspace(
            bounds_x_2[0] - expand_bounds, bounds_x_2[1] + expand_bounds, 1000
        )
        x_1, x_2 = np.meshgrid(xlist, ylist)

        first_term = -20 * np.exp(-0.2 * np.sqrt(1 / 2 * (x_1 ** 2 + x_2 ** 2)))
        second_term = -np.exp(
            1 / self.dim * (np.cos(2 * np.pi * x_1) + np.cos(2 * np.pi * x_2))
        )
        z = -(first_term + second_term + 20 + np.exp(1))

        ax.contour(x_1, x_2, z, levels=None, alpha=0.3)

        # plot evaluated and extra points with enumeration
        for c in range(len(points)):
            tmp_x_1, tmp_x_2 = points[c][0], points[c][1]
            ax.scatter(tmp_x_1, tmp_x_2)
            ax.text(tmp_x_1 + 0.01, tmp_x_2 + 0.01, c + 1, fontsize=7)

        # plot constraints
        if len(self.domain.constraints) > 0:
            x = np.linspace(bounds_x_1[0], bounds_x_1[1], 400)
            y = np.linspace(bounds_x_2[0], bounds_x_2[1], 400)
            x_1, x_2 = np.meshgrid(x, y)
            for c in self.domain.constraints:
                z = eval(c.lhs)
                ax.contour(x_1, x_2, z, [0], colors="grey", linestyles="dashed")

        # plot polygons
        if polygons:
            patches = []
            for i in range(len(polygons)):
                polygon_obj = Polygon(polygons[i], True, hatch="x")
                patches.append(polygon_obj)

            p = PatchCollection(patches, facecolors="None", edgecolors="grey", alpha=1)
            ax.add_collection(p)

        if return_fig:
            return fig, ax
        else:
            return ax

    def reset(self):
        super().reset()
        self.evaluated_points = []
