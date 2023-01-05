from typing import Optional

# import numpy as np
import pandas as pd

# from bofire.domain import Domain
# from bofire.domain.features import (
#     CategoricalDescriptorInput,
#     CategoricalInput,
#     ContinuousInput,
#     ContinuousOutput,
#     InputFeatures,
#     OutputFeatures,
# )
# from bofire.domain.objectives import MaximizeObjective, MinimizeObjective
from bofire.utils.study import Study

# from pydantic.types import PositiveInt



class SingleObjective(Study):
    def __init__(self, **data):
        super().__init__(**data)
        if (
            len(self.domain.output_features.get_by_objective(excludes=None)) > 1  # type: ignore
        ):  # TODO: update, when more features without DesFunc are implemented!
            raise ValueError("received multiobjective domain.")

    # TODO maybe unite with get_fbest from sobo, but not every strategy has get_fbest so far
    # and we have no universal way to compute it in domain --> maybe implement it also there.
    def get_fbest(self, experiments: Optional[pd.DataFrame] = None):
        if experiments is None:
            experiments = self.experiments
        ofeat = self.domain.output_features.get_by_objective(excludes=None)[0]  # type: ignore
        desirability = ofeat.desirability_function(experiments[ofeat.key])  # type: ignore
        return experiments.at[desirability.argmax(), ofeat.key]  # type: ignore

