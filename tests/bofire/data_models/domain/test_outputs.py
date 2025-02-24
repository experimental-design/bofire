import numpy as np
import pandas as pd
import pytest
from numpy import nan
from pandas.testing import assert_frame_equal

import tests.bofire.data_models.specs.api as specs
from bofire.data_models.domain.api import Outputs
from bofire.data_models.features.api import CategoricalOutput, ContinuousOutput
from bofire.data_models.objectives.api import (
    ConstrainedCategoricalObjective,
    ConstrainedObjective,
    MaximizeObjective,
    MaximizeSigmoidObjective,
    MovingMaximizeSigmoidObjective,
    Objective,
    TargetObjective,
)


data = pd.DataFrame.from_dict(
    {
        "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "x2": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "out1": [nan, 1.0, 2.0, 3.0, nan, nan],
        "out2": [nan, 1.0, 2.0, 3.0, 4.0, 5.0],
        "valid_out1": [1, 0, 1, 1, 1, 1],
        "valid_out2": [1, 1, 0, 1, 1, 0],
    },
)

obj = TargetObjective(target_value=1, steepness=2, tolerance=3, w=0.5)


of1 = ContinuousOutput(key="out1", objective=obj)
of2 = ContinuousOutput(key="out2", objective=obj)

of1_ = ContinuousOutput(key="out3", objective=None)
of2_ = ContinuousOutput(key="out4", objective=None)


@pytest.mark.parametrize(
    "outputs, data, output_feature_keys, expected",
    [
        (
            Outputs(features=[of1, of2]),
            data,
            None,
            pd.DataFrame.from_dict(
                {
                    "x1": [4.0],
                    "x2": [4.0],
                    "out1": [3.0],
                    "out2": [3.0],
                    "valid_out1": [1],
                    "valid_out2": [1],
                },
            ),
        ),
        (
            Outputs(features=[of1, of2]),
            data,
            [],
            pd.DataFrame.from_dict(
                {
                    "x1": [4.0],
                    "x2": [4.0],
                    "out1": [3.0],
                    "out2": [3.0],
                    "valid_out1": [1],
                    "valid_out2": [1],
                },
            ),
        ),
        (
            Outputs(features=[of1, of2]),
            data,
            ["out1", "out2"],
            pd.DataFrame.from_dict(
                {
                    "x1": [4.0],
                    "x2": [4.0],
                    "out1": [3.0],
                    "out2": [3.0],
                    "valid_out1": [1],
                    "valid_out2": [1],
                },
            ),
        ),
        (
            Outputs(features=[of1, of2]),
            data,
            ["out2"],
            pd.DataFrame.from_dict(
                {
                    "x1": [2.0, 4.0, 5.0],
                    "x2": [2.0, 4.0, 5.0],
                    "out1": [1.0, 3.0, nan],
                    "out2": [1.0, 3.0, 4.0],
                    "valid_out1": [0, 1, 1],
                    "valid_out2": [1, 1, 1],
                },
            ),
        ),
    ],
)
def test_preprocess_experiments_all_valid_outputs(
    outputs,
    data,
    output_feature_keys,
    expected,
):
    experiments = outputs.preprocess_experiments_all_valid_outputs(
        data,
        output_feature_keys,
    )
    assert_frame_equal(experiments.reset_index(drop=True), expected, check_dtype=False)


@pytest.mark.parametrize(
    "outputs, data, expected",
    [
        (
            Outputs(features=[of1, of2]),
            data,
            pd.DataFrame.from_dict(
                {
                    "x1": [2, 3, 4, 5],
                    "x2": [2, 3, 4, 5],
                    "out1": [1, 2, 3, nan],
                    "out2": [1, 2, 3, 4],
                    "valid_out1": [0, 1, 1, 1],
                    "valid_out2": [1, 0, 1, 1],
                },
            ),
        ),
    ],
)
def test_preprocess_experiments_any_valid_output(outputs, data, expected):
    experiments = outputs.preprocess_experiments_any_valid_output(data)
    assert experiments["x1"].tolist() == expected["x1"].tolist()
    assert experiments["out2"].tolist() == expected["out2"].tolist()


@pytest.mark.parametrize(
    "outputs, data, expected",
    [
        (
            Outputs(features=[of1, of2]),
            data,
            pd.DataFrame.from_dict(
                {
                    "x1": [2, 4, 5],
                    "x2": [2, 4, 5],
                    "out1": [1, 3, nan],
                    "out2": [1, 3, 4],
                    "valid_out1": [0, 1, 1],
                    "valid_out2": [1, 1, 1],
                },
            ),
        ),
    ],
)
def test_preprocess_experiments_one_valid_output(outputs, data, expected):
    experiments = outputs.preprocess_experiments_one_valid_output("out2", data)
    assert experiments["x1"].tolist() == expected["x1"].tolist()
    assert np.isnan(experiments["out1"].tolist()[2])
    assert experiments["out2"].tolist() == expected["out2"].tolist()


@pytest.mark.parametrize(
    "outputs, includes, excludes, exact, expected",
    [
        (Outputs(features=[of1, of2, of1_, of2_]), [Objective], [], True, []),
        (Outputs(features=[of1, of2, of1_, of2_]), [Objective], [], False, [of1, of2]),
        (
            Outputs(features=[of1, of2, of1_, of2_]),
            [TargetObjective],
            [],
            False,
            [of1, of2],
        ),
        (
            Outputs(features=[of1, of2, of1_, of2_]),
            [],
            [Objective],
            False,
            [of1_, of2_],
        ),
    ],
)
def test_get_outputs_by_objective(
    outputs: Outputs,
    includes,
    excludes,
    exact,
    expected,
):
    assert (
        outputs.get_by_objective(
            includes=includes,
            excludes=excludes,
            exact=exact,
        ).features
        == expected
    )


def test_get_outputs_by_objective_none():
    outputs = Outputs(
        features=[
            ContinuousOutput(key="a", objective=None),
            ContinuousOutput(
                key="b",
                objective=MaximizeSigmoidObjective(w=1, steepness=1, tp=0),
            ),
            ContinuousOutput(key="c", objective=MaximizeObjective()),
        ],
    )
    keys = outputs.get_keys_by_objective(excludes=ConstrainedObjective)
    assert keys == ["c"]
    assert outputs.get_keys().index("c") == 2
    assert outputs.get_keys_by_objective(excludes=Objective, includes=[]) == ["a"]
    assert outputs.get_by_objective(excludes=Objective, includes=[]) == Outputs(
        features=[ContinuousOutput(key="a", objective=None)],
    )


of1 = specs.features.valid(ContinuousOutput).obj(key="of1")
of2 = specs.features.valid(ContinuousOutput).obj(key="of2")
of3 = specs.features.valid(ContinuousOutput).obj(key="of3", objective=None)

outputs = Outputs(features=[of1, of2])

mixed_data = pd.DataFrame(
    columns=["of1", "of2", "of3"],
    index=range(5),
    data=np.random.uniform(size=(5, 3)),
)
mixed_data["of4"] = ["a", "a", "b", "b", "a"]


@pytest.mark.parametrize(
    "features, samples",
    [
        (
            outputs,
            pd.DataFrame(
                columns=["of1", "of2"],
                index=range(5),
                data=np.random.uniform(size=(5, 2)),
            ),
        ),
        (
            Outputs(features=[of1, of2, of3]),
            pd.DataFrame(
                columns=["of1", "of2", "of3"],
                index=range(5),
                data=np.random.uniform(size=(5, 3)),
            ),
        ),
        (
            Outputs(
                features=[
                    of1,
                    of2,
                    of3,
                    CategoricalOutput(
                        key="of4",
                        categories=["a", "b"],
                        objective=ConstrainedCategoricalObjective(
                            categories=["a", "b"],
                            desirability=[True, False],
                        ),
                    ),
                ],
            ),
            mixed_data,
        ),
    ],
)
def test_outputs_call(features, samples):
    o = features(samples)
    assert o.shape == (
        len(samples),
        len(
            features.get_keys_by_objective(
                Objective,
                excludes=ConstrainedCategoricalObjective,
            ),
        )
        + len(features.get_keys(CategoricalOutput)),
    )
    assert list(o.columns) == [
        f"{key}_des"
        for key in features.get_keys_by_objective(
            Objective,
            excludes=ConstrainedCategoricalObjective,
        )
        + features.get_keys(CategoricalOutput)
    ]


def test_outputs_call_adapt_experiment():
    outputs = Outputs(
        features=[
            ContinuousOutput(key="of1", objective=MaximizeObjective()),
            ContinuousOutput(
                key="of2",
                objective=MovingMaximizeSigmoidObjective(tp=0, steepness=10, w=1.0),
            ),
        ],
    )
    candidates = pd.DataFrame(
        columns=["of1_pred", "of2_pred"], data=[[1.0, 5.0], [2.0, 5.0]]
    )

    experiments = pd.DataFrame(columns=["of1", "of2"], data=[[1.0, 5.0], [2.0, 6.0]])

    with pytest.raises(
        ValueError,
        match="If predictions are used, `experiments_adapt` has to be provided.",
    ):
        outputs(candidates, predictions=True)

    outputs(experiments)
    outputs(candidates, experiments_adapt=experiments, predictions=True)


def test_categorical_objective_methods():
    obj = ConstrainedCategoricalObjective(
        categories=["a", "b"],
        desirability=[True, False],
    )
    assert obj.to_dict() == {"a": True, "b": False}
    assert obj.to_dict_label() == {"a": 0, "b": 1}
    assert obj.from_dict_label() == {0: "a", 1: "b"}


def test_categorical_output_methods():
    outputs = Outputs(
        features=[
            of1,
            of2,
            of3,
            CategoricalOutput(
                key="of4",
                categories=["a", "b"],
                objective=ConstrainedCategoricalObjective(
                    categories=["a", "b"],
                    desirability=[True, False],
                ),
            ),
        ],
    )

    # Test the `get_keys_by_objective`
    assert outputs.get_keys_by_objective(
        includes=Objective,
        excludes=ConstrainedObjective,
    ) == ["of1", "of2"]
    assert outputs.get_keys_by_objective(
        includes=ConstrainedObjective,
        excludes=None,
    ) == ["of4"]
