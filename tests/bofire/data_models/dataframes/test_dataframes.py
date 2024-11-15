from bofire.data_models.dataframes.api import Candidates, Experiments
from bofire.data_models.domain.api import Domain
from bofire.data_models.features.api import (
    CategoricalInput,
    CategoricalOutput,
    ContinuousInput,
    ContinuousOutput,
)
from bofire.data_models.objectives.api import ConstrainedCategoricalObjective
from tests.bofire.data_models.specs.dataframes import specs as dataframe_spec


def test_experiments_to_pandas():
    experiments: Experiments = dataframe_spec.valid(Experiments).obj()
    df_experiments = experiments.to_pandas()
    assert len(df_experiments) == len(experiments)
    domain = Domain.from_lists(
        inputs=[
            ContinuousInput(key="a", bounds=(0, 2)),
            CategoricalInput(key="b", categories=["cat", "cat2"]),
        ],
        outputs=[
            ContinuousOutput(key="alpha"),
            CategoricalOutput(
                key="beta",
                categories=["cat", "cat2"],
                objective=ConstrainedCategoricalObjective(
                    categories=["cat", "cat2"],
                    desirability=[True, False],
                ),
            ),
        ],
    )
    experiments2 = Experiments.from_pandas(df_experiments, domain)
    assert experiments == experiments2


def test_candidates_to_pandas():
    candidates: Candidates = dataframe_spec.valid(Candidates).obj()
    df_candidates = candidates.to_pandas()
    assert len(df_candidates) == len(candidates)
    domain = Domain.from_lists(
        inputs=[
            ContinuousInput(key="a", bounds=(0, 2)),
            CategoricalInput(key="b", categories=["cat", "cat2"]),
        ],
        outputs=[
            ContinuousOutput(key="alpha"),
            CategoricalOutput(
                key="beta",
                categories=["cat", "cat2"],
                objective=ConstrainedCategoricalObjective(
                    categories=["cat", "cat2"],
                    desirability=[True, False],
                ),
            ),
        ],
    )
    candidates2 = Candidates.from_pandas(df_candidates, domain)
    assert candidates == candidates2
