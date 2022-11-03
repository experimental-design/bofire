#%%%
from opti.problems.datasets import Cake

from bofire.mappers.opti import problem2domain

c = Cake()

config = c.to_config()

domain = problem2domain(config=config)

# %%
