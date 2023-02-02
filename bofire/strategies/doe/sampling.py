from itertools import combinations, product

import numpy as np
import opti


class Sampling:
    """Base class for sampling initial values for minimize_ipopt"""

    def __init__(self, problem: opti.Problem) -> None:
        """
        Args:
            problem (opti.Problem): problem defining the design space to sample from
        """
        self.problem = problem

    def sample(self, n_experiments: int) -> np.ndarray:
        raise NotImplementedError


class OptiSampling(Sampling):
    """Sampling using the sampling method from opti corresponding to the given problem."""

    def __init__(self, problem: opti.Problem) -> None:
        """
        Args:
            problem (opti.Problem): problem defining the design space to sample from
        """
        super().__init__(problem)

    def sample(self, n_experiments):
        """
        Args:
            n_experiments (int): Number of samples to draw.

        Returns:
            A numpy.ndarray, flattened version of the drawn samples.
        """
        return self.problem.sample_inputs(n_experiments).to_numpy().flatten()


class CornerSampling(Sampling):
    """Sampling from the corner points of the hypercubical domain defined by the inputs' bounds"""

    def __init__(self, problem: opti.Problem) -> None:
        """
        Args:
            problem (opti.Problem): problem defining the design space to sample from
        """
        super().__init__(problem)

    def sample(self, n_experiments) -> np.ndarray:
        """
        Args:
            n_experiments (int): Number of samples to draw.

        Returns:
            A numpy.ndarray, flattened version of the drawn samples.
        """
        D = self.problem.n_inputs

        # get all cornerpoints
        corners = np.array(list(product(*[[0, 1] for i in range(D)])))
        np.random.shuffle(corners)

        x0 = np.zeros(shape=(n_experiments, D))

        # draw samples
        for i in range(n_experiments):
            x0[i, :] = corners[i % len(corners)]

        return x0.flatten()


class ProbabilitySimplexSampling(Sampling):
    """Sampling from simplices that are derived from probability simplices by scaling axes with a constant positive factor"""

    def __init__(self, problem: opti.Problem) -> None:
        """
        Args:
            problem (opti.Problem): problem defining the design space to sample from
        """
        super().__init__(problem)
        self.check_problem_bounds()

    def check_problem_bounds(self) -> None:
        lb = self.problem.inputs.bounds.loc["min", :].to_numpy()
        if not np.allclose(lb, 0):
            raise ValueError(
                "problem has invalid bounds for ProbabilitySimplexSampling"
            )

    def sample(
        self, n_experiments: int, n_nonzero_components: int = None
    ) -> np.ndarray:
        """
        First samples from the corner points of the simplex, then random points where all but n_nonzero_components vanish.
        If the upper bounds are not equal for all inputs the resulting distribution will not be uniform, but the density is higher
        where the upper bounds are lower.

        Args:
            n_experiments (int): Number of samples to draw.
            n_nonzero_components (int): Number of nonzero components of the samples. Allows for sampling from the boundary only

        Returns:
            A numpy.ndarray, flattened version of the drawn samples.
        """

        D = self.problem.n_inputs
        if n_nonzero_components is None:
            n_nonzero_components = D

        x0 = np.zeros(shape=(n_experiments, D))

        # corner points
        corners = np.random.permutation(np.arange(D))

        nonzero_components = np.array(
            [c for c in combinations(np.arange(D), r=n_nonzero_components)]
        )
        np.random.shuffle(nonzero_components)

        # sample points
        for i in range(n_experiments):
            ind_nonzero_components = i % len(nonzero_components)

            if i < D:
                x0[i, corners[i]] = 1
            else:
                values_row = np.random.rand(n_nonzero_components)
                values_row /= np.linalg.norm(values_row, 1)
                x0[
                    np.repeat([i], n_nonzero_components),
                    nonzero_components[ind_nonzero_components],
                ] = values_row

        ub = self.problem.inputs.bounds.loc["max", :].to_numpy()
        for (i, bound) in enumerate(ub):
            x0[:, i] *= bound

        # shuffle points
        np.random.shuffle(x0)

        return x0.flatten()


# TODO: Mixed Sampling
