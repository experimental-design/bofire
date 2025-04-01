import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.repair import Repair
from pymoo.operators.crossover.hux import HUX
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize
from pymoo.problems.single.knapsack import create_random_knapsack_problem


problem = create_random_knapsack_problem(30)


class ConsiderMaximumWeightRepair(Repair):
    def _do(self, problem, Z, **kwargs):
        print(f"repair function called with {Z.shape[0]} individuals")
        # maximum capacity for the problem
        Q = problem.C

        # the corresponding weight of each individual
        weights = (Z * problem.W).sum(axis=1)

        # now repair each indvidiual i
        for i in range(len(Z)):
            # the packing plan for i
            z = Z[i]

            # while the maximum capacity violation holds
            while weights[i] > Q:
                # randomly select an item currently picked
                item_to_remove = np.random.choice(np.where(z)[0])

                # and remove it
                z[item_to_remove] = False

                # adjust the weight
                weights[i] -= problem.W[item_to_remove]

        return Z


algorithm = GA(
    pop_size=200,
    sampling=BinaryRandomSampling(),
    crossover=HUX(),
    mutation=BitflipMutation(),
    repair=ConsiderMaximumWeightRepair(),
    eliminate_duplicates=True,
)


res = minimize(problem, algorithm, termination=("n_gen", 10), verbose=True)
