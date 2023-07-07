from __future__ import annotations

from functools import total_ordering

import pandas as pd


@total_ordering
class NodeExperiment:
    def __init__(self, experiments: pd.DataFrame, value: float):
        self.experiments = experiments
        self.value = value

    def get_next_fixed_experiment(self):
        return pd.DataFrame(0)

    def __eq__(self, other: NodeExperiment):
        return self.value == other.value

    def __ne__(self, other: NodeExperiment):
        return self.value != other.value

    def __lt__(self, other: NodeExperiment):
        return self.value < other.value


def bnb(pr):
    pass
    # branch current solutions in sub-problems
    # solve branched problems
    # compare value with current lower bound
    # test if current solution is already valid
    #
