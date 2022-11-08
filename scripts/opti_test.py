# %%%

from opti import Problem
from opti.problems.datasets import Cake

from bofire.mappers.opti import domain2problem, problem2domain

c = Cake()

config = c.to_config()

domain = problem2domain(config=config)

config2 = domain2problem(domain, name="Cake")

c2 = Problem.from_config(config2)


# %%
