import pandas as pd
import pytest
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    StratifiedKFold,
)

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
                descriptors=["alpha", "beta"],
                values=[[1, 3], [2, 2], [3, 1]],
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
            input_preprocessing_specs={"x_3": CategoricalEncodingEnum.ORDINAL},
            categorical_encodings={"x_3": encoding},
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
                descriptors=["alpha", "beta"],
                values=[[1, 3], [2, 2], [3, 1]],
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
                descriptors=["alpha", "beta"],
                values=[[1, 3], [2, 2], [3, 1]],
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
                descriptors=["alpha", "beta"],
                values=[[1, 3], [2, 2], [3, 1]],
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
                descriptors=["alpha", "beta"],
                values=[[1, 3], [2, 2], [3, 1]],
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

    # Test GroupKFold split (default for group_split_column)
    cv, cv_func = model._make_cv_split(
        experiments,
        folds=2,
        random_state=1,
        stratified_feature=None,
        group_split_column="group",
        use_shuffle_split=False,  # Use GroupKFold (default)
    )
    assert isinstance(cv, GroupKFold)
    assert len(list(cv_func)) == 2

    # Test GroupShuffleSplit split (optional)
    cv, cv_func = model._make_cv_split(
        experiments,
        folds=2,
        random_state=1,
        stratified_feature=None,
        group_split_column="group",
        use_shuffle_split=True,  # Use GroupShuffleSplit
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


def test_model_cross_validate_timeseries():
    """Test cross-validation with timeseries data using GroupKFold."""
    # Create inputs with a timeseries feature
    inputs = Inputs(
        features=[
            ContinuousInput(
                key="time",
                bounds=(0, 100),
                is_timeseries=True,  # Mark as timeseries
            ),
            ContinuousInput(
                key="x",
                bounds=(-4, 4),
            ),
        ],
    )
    outputs = Outputs(features=[ContinuousOutput(key="y")])

    # Create experiments with overlapping time values across trajectories with realistic noise
    # This reflects realistic timeseries where different experiments run over similar time ranges
    # with slight measurement variations
    experiments = pd.DataFrame(
        {
            "time": [
                0.0,
                4.95,
                10.1,
                14.9,  # trajectory 0
                0.05,
                5.02,
                9.98,
                15.05,  # trajectory 1
                0.0,
                5.1,
                10.03,
                15.01,  # trajectory 2
                0.02,
                4.99,
                9.95,
                14.98,  # trajectory 3
            ],
            "x": [-4, -3, -2, -1, 0, 1, 2, 3, -4, -3, -2, -1, 0, 1, 2, 3],
            "y": [1, 2, 3, 4, 5, 6, 7, 8, 2, 3, 4, 5, 6, 7, 8, 9],
            "_trajectory_id": [
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
                3,
                3,
                3,
                3,
            ],  # Trajectory/experiment groups
            "valid_y": [1] * 16,
        }
    )

    # Initialize model
    model = SingleTaskGPSurrogate(
        inputs=inputs,
        outputs=outputs,
    )
    model = surrogates.map(model)

    # Test with automatic group_split_column for timeseries data
    # Since we have a timeseries feature and _trajectory_id column exists,
    # it should be used automatically
    train_cv, test_cv, _ = model.cross_validate(
        experiments,
        folds=4,
        # No group_split_column specified - should auto-use _trajectory_id
    )

    # Verify that groups are kept together
    test_indices = []
    train_indices = []
    for cvresults in test_cv.results:
        test_indices.append(list(cvresults.observed.index))

    for cvresults in train_cv.results:
        train_indices.append(list(cvresults.observed.index))

    # Get indices for each experiment group
    exp0_indices = experiments[experiments["_trajectory_id"] == 0].index.tolist()
    exp1_indices = experiments[experiments["_trajectory_id"] == 1].index.tolist()
    exp2_indices = experiments[experiments["_trajectory_id"] == 2].index.tolist()
    exp3_indices = experiments[experiments["_trajectory_id"] == 3].index.tolist()

    all_exp_indices = [exp0_indices, exp1_indices, exp2_indices, exp3_indices]

    # Verify that each experiment group is entirely in either train or test set
    for test_index, train_index in zip(test_indices, train_indices):
        for exp_indices in all_exp_indices:
            test_set = set(test_index)
            train_set = set(train_index)
            # Each experiment should be entirely in either test or train set
            assert test_set.issuperset(exp_indices) or train_set.issuperset(exp_indices)


def test_model_cross_validate_groupkfold_vs_shufflesplit():
    """Test that GroupKFold is default and GroupShuffleSplit can be enabled with flag."""
    inputs = Inputs(
        features=[
            ContinuousInput(
                key="time",
                bounds=(0, 100),
                is_timeseries=True,  # Timeseries feature
            ),
            ContinuousInput(key="x", bounds=(-4, 4)),
        ],
    )
    outputs = Outputs(features=[ContinuousOutput(key="y")])

    experiments = pd.DataFrame(
        {
            "time": [0, 5, 10, 15],
            "x": [-4, -3, -2, -1],
            "y": [1, 2, 3, 4],
            "group": [0, 0, 1, 1],
            "valid_y": [1] * 4,
        }
    )

    model = SingleTaskGPSurrogate(inputs=inputs, outputs=outputs)
    model = surrogates.map(model)

    # Test 1: Default behavior - should use GroupKFold
    cv_default, _ = model._make_cv_split(
        experiments,
        folds=2,
        group_split_column="group",
    )
    assert isinstance(cv_default, GroupKFold)

    # Test 2: With use_shuffle_split=True - should use GroupShuffleSplit
    cv_shuffle, _ = model._make_cv_split(
        experiments,
        folds=2,
        group_split_column="group",
        use_shuffle_split=True,
        random_state=42,
    )
    assert isinstance(cv_shuffle, GroupShuffleSplit)


def test_model_cross_validate_timeseries_automatic_trajectory_id():
    """Test that cross_validate automatically uses _trajectory_id column for timeseries."""
    # Create inputs with a timeseries feature
    inputs = Inputs(
        features=[
            ContinuousInput(
                key="time",
                bounds=(0, 100),
                is_timeseries=True,
            ),
            ContinuousInput(
                key="x",
                bounds=(-4, 4),
            ),
        ],
    )
    outputs = Outputs(features=[ContinuousOutput(key="y")])

    # Create experiments WITH a _trajectory_id column
    experiments = pd.DataFrame(
        {
            "_trajectory_id": [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
            "time": [0, 5, 10, 0, 5, 10, 0, 5, 10, 0, 5, 10],
            "x": [-4, -3, -2, -1, 0, 1, 2, 3, 4, -4, -3, -2],
            "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "valid_y": [1] * 12,
        }
    )

    model = SingleTaskGPSurrogate(
        inputs=inputs,
        outputs=outputs,
    )
    model = surrogates.map(model)

    # This should automatically use _trajectory_id as group_split_column
    cv_train, cv_test, _ = model.cross_validate(
        experiments,
        folds=2,
        include_X=True,  # Include X to check trajectory groups
        # No group_split_column specified - should auto-use _trajectory_id
    )

    # Check that results are returned
    assert len(cv_train.results) == 2
    assert len(cv_test.results) == 2

    # CRITICAL: Verify that trajectory groups are never split between train and test
    for fold_idx in range(2):
        train_data = cv_train.results[fold_idx].X
        test_data = cv_test.results[fold_idx].X

        # Get the trajectory IDs from the original experiments
        train_indices = train_data.index
        test_indices = test_data.index

        train_trajectories = set(experiments.loc[train_indices, "_trajectory_id"])
        test_trajectories = set(experiments.loc[test_indices, "_trajectory_id"])

        # Ensure no overlap between train and test trajectory groups
        assert (
            len(train_trajectories.intersection(test_trajectories)) == 0
        ), f"Fold {fold_idx}: Trajectory groups are mixed between train and test!"

        # Ensure all trajectories are accounted for
        assert train_trajectories.union(test_trajectories) == {
            0,
            1,
            2,
            3,
        }, f"Fold {fold_idx}: Not all trajectories are covered!"


def test_model_cross_validate_timeseries_use_shuffle_split():
    """Test that use_shuffle_split parameter works correctly with timeseries data."""
    inputs = Inputs(
        features=[
            ContinuousInput(key="time", bounds=(0, 100), is_timeseries=True),
            ContinuousInput(key="x", bounds=(-4, 4)),
        ],
    )
    outputs = Outputs(features=[ContinuousOutput(key="y")])

    # Create experiments with 4 trajectories
    experiments = pd.DataFrame(
        {
            "_trajectory_id": [0, 0, 1, 1, 2, 2, 3, 3],
            "time": [0, 10, 0, 10, 0, 10, 0, 10],
            "x": [-4, -3, -2, -1, 0, 1, 2, 3],
            "y": [1, 2, 3, 4, 5, 6, 7, 8],
            "valid_y": [1] * 8,
        }
    )

    model = SingleTaskGPSurrogate(inputs=inputs, outputs=outputs)
    model = surrogates.map(model)

    # Test with GroupKFold (default) - should use all trajectories
    train_cv_kfold, test_cv_kfold, _ = model.cross_validate(
        experiments,
        folds=2,
    )

    # Test with GroupShuffleSplit - may not use all trajectories
    train_cv_shuffle, test_cv_shuffle, _ = model.cross_validate(
        experiments,
        folds=2,
        use_shuffle_split=True,
        random_state=42,
    )

    # Both should return results
    assert len(train_cv_kfold.results) == 2
    assert len(test_cv_kfold.results) == 2
    assert len(train_cv_shuffle.results) == 2
    assert len(test_cv_shuffle.results) == 2


def test_model_cross_validate_groupkfold_exhaustive():
    """Test that GroupKFold ensures every trajectory is tested exactly once."""
    inputs = Inputs(
        features=[
            ContinuousInput(key="time", bounds=(0, 100), is_timeseries=True),
            ContinuousInput(key="x", bounds=(-4, 4)),
        ],
    )
    outputs = Outputs(features=[ContinuousOutput(key="y")])

    # Create experiments with 6 trajectories
    n_trajectories = 6
    experiments = pd.DataFrame(
        {
            "_trajectory_id": [i for i in range(n_trajectories) for _ in range(3)],
            "time": [0, 5, 10] * n_trajectories,
            "x": list(range(18)),
            "y": list(range(18)),
            "valid_y": [1] * 18,
        }
    )

    model = SingleTaskGPSurrogate(inputs=inputs, outputs=outputs)
    model = surrogates.map(model)

    # Use 3 folds - each trajectory should appear in exactly one test fold
    train_cv, test_cv, _ = model.cross_validate(
        experiments,
        folds=3,
    )

    # Collect all tested trajectory IDs across folds
    tested_trajectories = set()
    for cv_result in test_cv.results:
        test_indices = cv_result.observed.index
        test_data = experiments.iloc[test_indices]
        tested_trajectories.update(test_data["_trajectory_id"].unique())

    # Every trajectory should be tested exactly once
    assert tested_trajectories == set(range(n_trajectories))

    # Check that no trajectory appears in multiple test sets
    trajectory_test_counts = dict.fromkeys(range(n_trajectories), 0)
    for cv_result in test_cv.results:
        test_indices = cv_result.observed.index
        test_data = experiments.iloc[test_indices]
        for traj_id in test_data["_trajectory_id"].unique():
            trajectory_test_counts[traj_id] += 1

    # Each trajectory should be tested exactly once
    assert all(count == 1 for count in trajectory_test_counts.values())


def test_model_cross_validate_timeseries_auto_infer_no_trajectories():
    """Test that error is raised when auto-inference finds no valid trajectories."""
    # Create inputs with a timeseries feature
    inputs = Inputs(
        features=[
            ContinuousInput(
                key="time",
                bounds=(0, 100),
                is_timeseries=True,
            ),
            ContinuousInput(
                key="x",
                bounds=(-4, 4),
            ),
        ],
    )
    outputs = Outputs(features=[ContinuousOutput(key="y")])

    # Create experiments WITHOUT the _trajectory_id column
    # All x values are unique, so no trajectories can be inferred
    experiments = pd.DataFrame(
        {
            "time": [0, 5, 10, 15],
            "x": [-4, -3, -2, -1],  # All unique - no valid trajectories
            "y": [1, 2, 3, 4],
            "valid_y": [1] * 4,
        }
    )

    model = SingleTaskGPSurrogate(
        inputs=inputs,
        outputs=outputs,
    )
    model = surrogates.map(model)

    # Should raise error about no valid trajectories found
    with pytest.raises(
        ValueError,
        match="No valid timeseries trajectories found",
    ):
        model.cross_validate(
            experiments,
            folds=2,
        )


def test_model_cross_validate_timeseries_auto_infer_success():
    """Test that trajectory IDs are auto-inferred when _trajectory_id column is missing."""
    # Create inputs with a timeseries feature
    inputs = Inputs(
        features=[
            ContinuousInput(
                key="time",
                bounds=(0, 100),
                is_timeseries=True,
            ),
            ContinuousInput(
                key="temperature",
                bounds=(20, 80),
            ),
        ],
    )
    outputs = Outputs(features=[ContinuousOutput(key="y")])

    # Create experiments WITHOUT the _trajectory_id column
    # Temperature has repeated values, so trajectories can be inferred
    experiments = pd.DataFrame(
        {
            "time": [0, 5, 10, 0, 5, 10, 0, 5, 10],
            "temperature": [25, 25, 25, 30, 30, 30, 35, 35, 35],  # 3 trajectories
            "y": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "valid_y": [1] * 9,
        }
    )

    model = SingleTaskGPSurrogate(
        inputs=inputs,
        outputs=outputs,
    )
    model = surrogates.map(model)

    # Should warn about auto-inference and succeed
    with pytest.warns(
        UserWarning,
        match="Timeseries feature 'time' detected but '_trajectory_id' column is missing",
    ):
        train_cv, test_cv, _ = model.cross_validate(
            experiments,
            folds=3,
        )

    # Verify results
    assert len(train_cv.results) == 3
    assert len(test_cv.results) == 3

    # Verify trajectories are kept together (each fold tests one temperature group)
    for cv_result in test_cv.results:
        test_indices = cv_result.observed.index
        test_temps = experiments.loc[test_indices, "temperature"].unique()
        # Each fold should test exactly one temperature group
        assert len(test_temps) == 1


def test_model_cross_validate_trajectory_id_without_timeseries():
    """Test that _trajectory_id column triggers group split even without is_timeseries feature."""
    # Create inputs WITHOUT any timeseries feature
    inputs = Inputs(
        features=[
            ContinuousInput(key="x1", bounds=(-10, 10)),
            ContinuousInput(key="x2", bounds=(-10, 10)),
        ]
    )
    outputs = Outputs(features=[ContinuousOutput(key="y")])

    # Create experiments with _trajectory_id column
    experiments = pd.DataFrame(
        {
            "_trajectory_id": [0, 0, 0, 1, 1, 1, 2, 2, 2],
            "x1": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "x2": [9, 8, 7, 6, 5, 4, 3, 2, 1],
            "y": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "valid_y": [1] * 9,
        }
    )

    model = SingleTaskGPSurrogate(
        inputs=inputs,
        outputs=outputs,
    )
    model = surrogates.map(model)

    # Cross-validate without specifying group_split_column
    # Should automatically use _trajectory_id
    _, test_cv, _ = model.cross_validate(experiments, folds=3)

    # Verify groups are kept together
    for cv_result in test_cv.results:
        test_indices = cv_result.observed.index
        test_trajectories = experiments.loc[test_indices, "_trajectory_id"].unique()
        # Each fold should test exactly one trajectory
        assert (
            len(test_trajectories) == 1
        ), f"Expected 1 trajectory per fold, got {len(test_trajectories)}: {test_trajectories}"
