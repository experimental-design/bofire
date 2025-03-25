import pandas as pd
import pytest
from sklearn.model_selection import GroupShuffleSplit, KFold, StratifiedKFold

import bofire.surrogates.api as surrogates
from bofire.data_models.domain.api import Inputs, Outputs
from bofire.data_models.enum import CategoricalEncodingEnum
from bofire.data_models.features.api import (
    CategoricalDescriptorInput,
    CategoricalInput,
    ContinuousInput,
    ContinuousOutput,
)
from bofire.data_models.surrogates.api import SingleTaskGPSurrogate


@pytest.mark.parametrize("folds", [5, 3, 10, -1])
def test_model_cross_validate(folds):
    inputs = Inputs(
        features=[
            ContinuousInput(
                key=f"x_{i+1}",
                bounds=(-4, 4),
            )
            for i in range(2)
        ],
    )
    outputs = Outputs(features=[ContinuousOutput(key="y")])
    experiments = inputs.sample(n=100)
    experiments.eval("y=((x_1**2 + x_2 - 11)**2+(x_1 + x_2**2 -7)**2)", inplace=True)
    experiments["valid_y"] = 1
    experiments = experiments.sample(10)
    model = SingleTaskGPSurrogate(
        inputs=inputs,
        outputs=outputs,
    )
    model = surrogates.map(model)
    train_cv, test_cv, _ = model.cross_validate(experiments, folds=folds)
    efolds = folds if folds != -1 else 10
    assert len(train_cv.results) == efolds
    assert len(test_cv.results) == efolds


def test_model_cross_validate_descriptor():
    folds = 5
    inputs = Inputs(
        features=[
            ContinuousInput(
                key=f"x_{i+1}",
                bounds=(-4, 4),
            )
            for i in range(2)
        ]
        + [
            CategoricalDescriptorInput(
                key="x_3",
                categories=["a", "b", "c"],
                descriptors=["alpha"],
                values=[[1], [2], [3]],
            ),
        ],
    )
    outputs = Outputs(features=[ContinuousOutput(key="y")])
    experiments = inputs.sample(n=100)
    experiments.eval("y=((x_1**2 + x_2 - 11)**2+(x_1 + x_2**2 -7)**2)", inplace=True)
    experiments.loc[experiments.x_2 == "b", "y"] += 5
    experiments.loc[experiments.x_2 == "c", "y"] += 10
    experiments["valid_y"] = 1
    experiments = experiments.sample(10)
    for encoding in [
        CategoricalEncodingEnum.ONE_HOT,
        CategoricalEncodingEnum.DESCRIPTOR,
    ]:
        model = SingleTaskGPSurrogate(
            inputs=inputs,
            outputs=outputs,
            input_preprocessing_specs={"x_3": encoding},
        )
        model = surrogates.map(model)
        train_cv, test_cv, _ = model.cross_validate(experiments, folds=folds)
        efolds = folds if folds != -1 else 10
        assert len(train_cv.results) == efolds
        assert len(test_cv.results) == efolds


@pytest.mark.parametrize("include_X, include_labcodes", [[True, False], [False, True]])
def test_model_cross_validate_include_X(include_X, include_labcodes):
    inputs = Inputs(
        features=[
            ContinuousInput(
                key=f"x_{i+1}",
                bounds=(-4, 4),
            )
            for i in range(2)
        ],
    )
    outputs = Outputs(features=[ContinuousOutput(key="y")])
    experiments = inputs.sample(n=10)
    experiments["labcode"] = [str(i) for i in range(10)]
    experiments.eval("y=((x_1**2 + x_2 - 11)**2+(x_1 + x_2**2 -7)**2)", inplace=True)
    experiments["valid_y"] = 1
    model = SingleTaskGPSurrogate(
        inputs=inputs,
        outputs=outputs,
    )
    model = surrogates.map(model)
    train_cv, test_cv, _ = model.cross_validate(
        experiments,
        folds=5,
        include_X=include_X,
        include_labcodes=include_labcodes,
    )
    if include_X:
        assert train_cv.results[0].X.shape == (8, 2)
        assert test_cv.results[0].X.shape == (2, 2)
    if include_X is False:
        assert train_cv.results[0].X is None
        assert test_cv.results[0].X is None
    if include_labcodes:
        assert train_cv.results[0].labcodes.shape == (8,)
        assert test_cv.results[0].labcodes.shape == (2,)
    else:
        assert train_cv.results[0].labcodes is None
        assert train_cv.results[0].labcodes is None


def test_model_cross_validate_hooks():
    def hook1(surrogate, X_train, y_train, X_test, y_test):
        assert isinstance(surrogate, surrogates.SingleTaskGPSurrogate)
        assert y_train.shape == (8, 1)
        assert y_test.shape == (2, 1)
        return X_train.shape

    def hook2(surrogate, X_train, y_train, X_test, y_test, return_test=True):
        if return_test:
            return X_test.shape
        return X_train.shape

    inputs = Inputs(
        features=[
            ContinuousInput(
                key=f"x_{i+1}",
                bounds=(-4, 4),
            )
            for i in range(2)
        ],
    )
    outputs = Outputs(features=[ContinuousOutput(key="y")])
    experiments = inputs.sample(n=10)
    experiments.eval("y=((x_1**2 + x_2 - 11)**2+(x_1 + x_2**2 -7)**2)", inplace=True)
    experiments["valid_y"] = 1
    model = SingleTaskGPSurrogate(
        inputs=inputs,
        outputs=outputs,
    )
    model = surrogates.map(model)
    # first test with one hook
    _, _, hook_results = model.cross_validate(
        experiments,
        folds=5,
        hooks={"hook1": hook1},
    )
    assert len(hook_results.keys()) == 1
    assert len(hook_results["hook1"]) == 5
    assert hook_results["hook1"] == [(8, 2), (8, 2), (8, 2), (8, 2), (8, 2)]
    # now test with two hooks
    _, _, hook_results = model.cross_validate(
        experiments,
        folds=5,
        hooks={"hook1": hook1, "hook2": hook2},
    )
    assert len(hook_results.keys()) == 2
    assert len(hook_results["hook1"]) == 5
    assert hook_results["hook1"] == [(8, 2), (8, 2), (8, 2), (8, 2), (8, 2)]
    assert len(hook_results["hook2"]) == 5
    assert hook_results["hook2"] == [(2, 2), (2, 2), (2, 2), (2, 2), (2, 2)]
    # now test with two hooks and keyword arguments
    _, _, hook_results = model.cross_validate(
        experiments,
        folds=5,
        hooks={"hook1": hook1, "hook2": hook2},
        hook_kwargs={"hook2": {"return_test": False}},
    )
    assert len(hook_results.keys()) == 2
    assert len(hook_results["hook1"]) == 5
    assert hook_results["hook1"] == [(8, 2), (8, 2), (8, 2), (8, 2), (8, 2)]
    assert len(hook_results["hook2"]) == 5
    assert hook_results["hook2"] == [(8, 2), (8, 2), (8, 2), (8, 2), (8, 2)]


@pytest.mark.parametrize("folds", [-2, 0, 1])
def test_model_cross_validate_invalid(folds):
    inputs = Inputs(
        features=[
            ContinuousInput(
                key=f"x_{i+1}",
                bounds=(-4, 4),
            )
            for i in range(2)
        ],
    )
    outputs = Outputs(features=[ContinuousOutput(key="y")])
    experiments = inputs.sample(n=10)
    experiments.eval("y=((x_1**2 + x_2 - 11)**2+(x_1 + x_2**2 -7)**2)", inplace=True)
    experiments["valid_y"] = 1
    model = SingleTaskGPSurrogate(
        inputs=inputs,
        outputs=outputs,
    )
    model = surrogates.map(model)
    with pytest.raises(ValueError, match="Folds must be -1 for LOO, or > 1."):
        model.cross_validate(experiments, folds=folds)


@pytest.mark.parametrize("folds", [5, 3, 10, -1])
def test_model_cross_validate_random_state(folds):
    inputs = Inputs(
        features=[
            ContinuousInput(
                key=f"x_{i+1}",
                bounds=(-4, 4),
            )
            for i in range(2)
        ],
    )
    outputs = Outputs(features=[ContinuousOutput(key="y")])
    experiments = inputs.sample(n=100)
    experiments.eval("y=((x_1**2 + x_2 - 11)**2+(x_1 + x_2**2 -7)**2)", inplace=True)
    experiments["valid_y"] = 1
    experiments = experiments.sample(10)
    model = SingleTaskGPSurrogate(
        inputs=inputs,
        outputs=outputs,
    )
    model = surrogates.map(model)
    train_cv_1_1, test_cv_1_1, _ = model.cross_validate(
        experiments,
        folds=folds,
        random_state=1,
    )
    train_cv_1_2, test_cv_1_2, _ = model.cross_validate(
        experiments,
        folds=folds,
        random_state=1,
    )

    train_cv_2_1, test_cv_2_1, _ = model.cross_validate(
        experiments,
        folds=folds,
        random_state=2,
    )
    train_cv_2_2, test_cv_2_2, _ = model.cross_validate(
        experiments,
        folds=folds,
        random_state=2,
    )

    for cvresult1, cvresult2 in zip(train_cv_1_1.results, train_cv_1_2.results):
        assert all(list(cvresult1.observed.index == cvresult2.observed.index))

    for cvresult1, cvresult2 in zip(test_cv_1_1.results, test_cv_1_2.results):
        assert all(list(cvresult1.observed.index == cvresult2.observed.index))

    for cvresult1, cvresult2 in zip(train_cv_2_1.results, train_cv_2_2.results):
        assert all(list(cvresult1.observed.index == cvresult2.observed.index))

    for cvresult1, cvresult2 in zip(test_cv_2_1.results, test_cv_2_2.results):
        assert all(list(cvresult1.observed.index == cvresult2.observed.index))

    for cvresult1, cvresult2 in zip(test_cv_1_1.results, test_cv_2_1.results):
        assert not all(list(cvresult1.observed.index == cvresult2.observed.index))

    for cvresult1, cvresult2 in zip(test_cv_1_2.results, test_cv_2_2.results):
        assert not all(list(cvresult1.observed.index == cvresult2.observed.index))


# Include test for CategoricalOutput when fully implemented
@pytest.mark.parametrize("random_state", [1, 2])
def test_model_cross_validate_stratified(random_state):
    inputs = Inputs(
        features=[
            ContinuousInput(
                key=f"x_{i+1}",
                bounds=(-4, 4),
            )
            for i in range(2)
        ]
        + [
            CategoricalInput(key="cat_x_3", categories=["category1", "category2"]),
            CategoricalDescriptorInput(
                key="cat_x_4",
                categories=["a", "b", "c"],
                descriptors=["alpha"],
                values=[[1], [2], [3]],
            ),
        ],
    )
    outputs = Outputs(features=[ContinuousOutput(key="y")])
    # category2, b, and c only appears 5 times each
    experiments = pd.DataFrame(
        [
            [-4, -4, "category1", "a", 1],
            [-3, -3, "category1", "a", 1],
            [-2, -2, "category1", "a", 1],
            [-1, -1, "category1", "b", 1],
            [0, 0, "category1", "b", 1],
            [1, 1, "category1", "b", 1],
            [2, 2, "category1", "c", 1],
            [3, 3, "category1", "c", 1],
            [2, 3, "category1", "c", 1],
            [3, 1, "category1", "a", 1],
            [3, 4, "category1", "a", 0],
            [4, 4, "category2", "b", 0],
            [1, 4, "category2", "b", 0],
            [1, 0, "category2", "c", 0],
            [1, 2, "category2", "c", 0],
            [2, 4, "category2", "a", 1],
        ],
        columns=["x_1", "x_2", "cat_x_3", "cat_x_4", "y"],
    )
    experiments["valid_y"] = 1

    cat_x_3_category2_indexes = experiments.index[
        experiments["cat_x_3"] == "category2"
    ].tolist()
    cat_x_4_b_indexes = experiments.index[experiments["cat_x_4"] == "b"].tolist()
    cat_x_4_c_indexes = experiments.index[experiments["cat_x_4"] == "c"].tolist()
    zero_indexes = experiments.index[experiments["y"] == 0].tolist()

    model = SingleTaskGPSurrogate(
        inputs=inputs,
        outputs=outputs,
    )
    model = surrogates.map(model)
    _, test_cv, _ = model.cross_validate(
        experiments,
        folds=5,
        random_state=random_state,
        stratified_feature="cat_x_3",
    )
    for cvresults in test_cv.results:
        assert any(i in cvresults.observed.index for i in cat_x_3_category2_indexes)

    _, test_cv, _ = model.cross_validate(
        experiments,
        folds=5,
        random_state=random_state,
        stratified_feature="cat_x_4",
    )
    for cvresults in test_cv.results:
        assert any(i in cvresults.observed.index for i in cat_x_4_b_indexes)
        assert any(i in cvresults.observed.index for i in cat_x_4_c_indexes)

    _, test_cv, _ = model.cross_validate(
        experiments,
        folds=5,
        random_state=random_state,
        stratified_feature="y",
    )
    for cvresults in test_cv.results:
        assert any(i in cvresults.observed.index for i in zero_indexes)


def test_model_cross_validate_stratified_invalid_feature_name():
    inputs = Inputs(
        features=[
            ContinuousInput(
                key=f"x_{i+1}",
                bounds=(-4, 4),
            )
            for i in range(2)
        ]
        + [
            CategoricalInput(key="cat_x_3", categories=["category1", "category2"]),
            CategoricalDescriptorInput(
                key="cat_x_4",
                categories=["a", "b", "c"],
                descriptors=["alpha"],
                values=[[1], [2], [3]],
            ),
        ],
    )
    outputs = Outputs(features=[ContinuousOutput(key="y")])
    experiments = pd.DataFrame(
        [
            [-4, -4, "category1", "a", 1],
            [-3, -3, "category1", "a", 1],
            [-2, -2, "category1", "a", 1],
            [-1, -1, "category1", "b", 1],
            [0, 0, "category1", "b", 1],
            [1, 1, "category1", "b", 1],
            [2, 2, "category1", "c", 1],
            [3, 3, "category1", "c", 1],
            [2, 3, "category1", "c", 1],
            [3, 1, "category1", "a", 1],
            [3, 4, "category1", "a", 0],
            [4, 4, "category2", "b", 0],
            [1, 4, "category2", "b", 0],
            [1, 0, "category2", "c", 0],
            [1, 2, "category2", "c", 0],
            [2, 4, "category2", "a", 1],
        ],
        columns=["x_1", "x_2", "cat_x_3", "cat_x_4", "y"],
    )
    experiments["valid_y"] = 1

    model = SingleTaskGPSurrogate(
        inputs=inputs,
        outputs=outputs,
    )
    model = surrogates.map(model)
    with pytest.raises(
        ValueError,
        match="The feature to be stratified is not in the model inputs or outputs",
    ):
        model.cross_validate(experiments, folds=5, stratified_feature="name")


@pytest.mark.parametrize("key", ["x_1", "x_2"])
def test_model_cross_validate_stratified_invalid_feature_type(key):
    inputs = Inputs(
        features=[
            ContinuousInput(
                key=f"x_{i+1}",
                bounds=(-4, 4),
            )
            for i in range(2)
        ]
        + [
            CategoricalInput(key="cat_x_3", categories=["category1", "category2"]),
            CategoricalDescriptorInput(
                key="cat_x_4",
                categories=["a", "b", "c"],
                descriptors=["alpha"],
                values=[[1], [2], [3]],
            ),
        ],
    )
    outputs = Outputs(features=[ContinuousOutput(key="y")])
    experiments = pd.DataFrame(
        [
            [-4, -4, "category1", "a", 1],
            [-3, -3, "category1", "a", 1],
            [-2, -2, "category1", "a", 1],
            [-1, -1, "category1", "b", 1],
            [0, 0, "category1", "b", 1],
            [1, 1, "category1", "b", 1],
            [2, 2, "category1", "c", 1],
            [3, 3, "category1", "c", 1],
            [2, 3, "category1", "c", 1],
            [3, 1, "category1", "a", 1],
            [3, 4, "category1", "a", 0],
            [4, 4, "category2", "b", 0],
            [1, 4, "category2", "b", 0],
            [1, 0, "category2", "c", 0],
            [1, 2, "category2", "c", 0],
            [2, 4, "category2", "a", 1],
        ],
        columns=["x_1", "x_2", "cat_x_3", "cat_x_4", "y"],
    )
    experiments["valid_y"] = 1

    model = SingleTaskGPSurrogate(
        inputs=inputs,
        outputs=outputs,
    )
    model = surrogates.map(model)
    with pytest.raises(
        ValueError,
        match="The feature to be stratified needs to be a DiscreteInput, CategoricalInput, CategoricalOutput, or ContinuousOutput",
    ):
        model.cross_validate(experiments, folds=5, stratified_feature=key)


@pytest.mark.parametrize("random_state", [1, 2])
def test_model_cross_validate_groupfold(random_state):
    # Define the input features for the model
    inputs = Inputs(
        features=[
            ContinuousInput(
                key=f"x_{i+1}",
                bounds=(-4, 4),
            )
            for i in range(2)
        ]
        + [
            CategoricalInput(key="cat_x_3", categories=["category1", "category2"]),
            CategoricalDescriptorInput(
                key="cat_x_4",
                categories=["a", "b", "c"],
                descriptors=["alpha"],
                values=[[1], [2], [3]],
            ),
        ],
    )
    # Define the output features for the model
    outputs = Outputs(features=[ContinuousOutput(key="y")])

    # Create a DataFrame with sample experiments data
    experiments = pd.DataFrame(
        [
            [-4, -4, "category1", "a", 1, 0],
            [-3, -3, "category1", "a", 1, 0],
            [-2, -2, "category1", "a", 1, 0],
            [-1, -1, "category1", "b", 1, 0],
            [0, 0, "category1", "b", 1, 1],
            [1, 1, "category1", "b", 1, 1],
            [2, 2, "category1", "c", 1, 1],
            [3, 3, "category1", "c", 1, 1],
            [2, 3, "category1", "c", 1, 1],
            [3, 1, "category1", "a", 1, 2],
            [3, 4, "category1", "a", 0, 2],
            [4, 4, "category2", "b", 0, 2],
            [1, 4, "category2", "b", 0, 2],
            [1, 0, "category2", "c", 0, 2],
            [1, 2, "category2", "c", 0, 3],
            [2, 4, "category2", "a", 1, 3],
        ],
        columns=["x_1", "x_2", "cat_x_3", "cat_x_4", "y", "group"],
    )
    experiments["valid_y"] = 1

    # Get the indices for each group
    cat0_indexes = experiments[experiments["group"] == 0].index
    cat1_indexes = experiments[experiments["group"] == 1].index
    cat2_indexes = experiments[experiments["group"] == 2].index
    cat3_indexes = experiments[experiments["group"] == 3].index

    all_indices = [cat0_indexes, cat1_indexes, cat2_indexes, cat3_indexes]

    # Initialize the model
    model = SingleTaskGPSurrogate(
        inputs=inputs,
        outputs=outputs,
    )
    model = surrogates.map(model)

    # Perform cross-validation with group splitting
    train_cv, test_cv, hook_results = model.cross_validate(
        experiments,
        folds=4,
        random_state=random_state,
        group_split_column="group",
    )

    # Gather train and test indices
    test_indices = []
    train_indices = []
    for cvresults in test_cv.results:
        test_indices.append(list(cvresults.observed.index))

    for cvresults in train_cv.results:
        train_indices.append(list(cvresults.observed.index))

    # Test if the groups are only present in either the test or train indices and are grouped together
    for test_index, train_index in zip(test_indices, train_indices):
        for indices in all_indices:
            test_set = set(test_index)
            train_set = set(train_index)
            assert test_set.issuperset(indices) or train_set.issuperset(indices)


def test_model_cross_validate_invalid_group_split_column():
    inputs = Inputs(
        features=[
            ContinuousInput(
                key=f"x_{i+1}",
                bounds=(-4, 4),
            )
            for i in range(2)
        ],
    )
    outputs = Outputs(features=[ContinuousOutput(key="y")])
    experiments = inputs.sample(n=10)
    experiments.eval("y=((x_1**2 + x_2 - 11)**2+(x_1 + x_2**2 -7)**2)", inplace=True)
    experiments["valid_y"] = 1
    model = SingleTaskGPSurrogate(
        inputs=inputs,
        outputs=outputs,
    )
    model = surrogates.map(model)

    # Test with a non-existent group split column
    with pytest.raises(
        ValueError,
        match="Group split column non_existent_column is not present in the experiments.",
    ):
        model.cross_validate(
            experiments, folds=5, group_split_column="non_existent_column"
        )

    # Test with fewer unique groups than folds
    experiments["group"] = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2]
    with pytest.raises(
        ValueError,
        match="Number of unique groups 3 is less than the number of folds 5.",
    ):
        model.cross_validate(experiments, folds=5, group_split_column="group")


def test_make_cv_split():
    inputs = Inputs(
        features=[
            ContinuousInput(
                key=f"x_{i+1}",
                bounds=(-4, 4),
            )
            for i in range(2)
        ],
    )
    outputs = Outputs(features=[ContinuousOutput(key="y")])
    experiments = inputs.sample(n=10)
    experiments["group"] = [i % 2 for i in range(10)]
    experiments["stratified_feature"] = [
        (i % 2) == 0 for i in range(10)
    ]  # Add a stratified feature
    experiments.eval("y=((x_1**2 + x_2 - 11)**2+(x_1 + x_2**2 -7)**2)", inplace=True)
    experiments["valid_y"] = 1
    model = SingleTaskGPSurrogate(
        inputs=inputs,
        outputs=outputs,
    )
    model = surrogates.map(model)

    # Test KFold split
    cv, cv_func = model._make_cv_split(
        experiments,
        folds=5,
        random_state=1,
        stratified_feature=None,
        group_split_column=None,
    )
    assert isinstance(cv, KFold)
    assert len(list(cv_func)) == 5

    # Test StratifiedKFold split
    cv, cv_func = model._make_cv_split(
        experiments,
        folds=5,
        random_state=1,
        stratified_feature="stratified_feature",
        group_split_column=None,
    )
    assert isinstance(cv, StratifiedKFold)
    assert len(list(cv_func)) == 5

    # Test GroupShuffleSplit split
    cv, cv_func = model._make_cv_split(
        experiments,
        folds=2,
        random_state=1,
        stratified_feature=None,
        group_split_column="group",
    )
    assert isinstance(cv, GroupShuffleSplit)
    assert len(list(cv_func)) == 2


def test_check_valid_nfolds():
    inputs = Inputs(
        features=[
            ContinuousInput(
                key=f"x_{i+1}",
                bounds=(-4, 4),
            )
            for i in range(2)
        ],
    )
    outputs = Outputs(features=[ContinuousOutput(key="y")])
    model = SingleTaskGPSurrogate(
        inputs=inputs,
        outputs=outputs,
    )
    model = surrogates.map(model)
    # Test valid folds
    assert model._check_valid_nfolds(5, 10) == 5
    assert model._check_valid_nfolds(-1, 10) == 10

    # Test folds greater than number of experiments
    with pytest.warns(
        UserWarning,
        match="Training data only has 10 experiments, which is less than folds, fallback to LOOCV.",
    ):
        assert model._check_valid_nfolds(20, 10) == 10

    # Test invalid folds
    with pytest.raises(ValueError, match="Folds must be -1 for LOO, or > 1."):
        model._check_valid_nfolds(0, 10)
    with pytest.raises(ValueError, match="Folds must be -1 for LOO, or > 1."):
        model._check_valid_nfolds(1, 10)
    with pytest.raises(ValueError, match="Experiments is empty."):
        model._check_valid_nfolds(5, 0)
