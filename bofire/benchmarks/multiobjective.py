from typing import Optional

import pandas as pd

from bofire.utils.multiobjective import compute_hypervolume, get_pareto_front
from bofire.utils.study import Study


class MultiObjective(Study):

    ref_point: Optional[dict]

    def get_fbest(self, experiments: Optional[pd.DataFrame] = None):
        if experiments is None:
            experiments = self.experiments
        optimal_experiments = get_pareto_front(self.domain, experiments)  # type: ignore
        return compute_hypervolume(self.domain, optimal_experiments, self.ref_point)  # type: ignore

    def __init__(self, **data):
        super().__init__(**data)
        if (
            len(self.domain.output_features.get_by_objective(excludes=None)) < 2  # type: ignore
        ):  # TODO: update, when more features without DesFunc are implemented!
            raise ValueError("received singelobjective domain.")
