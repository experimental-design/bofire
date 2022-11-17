from typing import List, Union

import numpy as np
from scipy import sparse
from sklearn.preprocessing._encoders import _BaseEncoder
from sklearn.utils import check_array
from sklearn.utils._encode import _check_unknown, _unique
from sklearn.utils.validation import _check_feature_names_in, check_is_fitted


class CategoricalDescriptorEncoder(_BaseEncoder):
    """
    Encoder to translate categorical parameters into continuous descriptor values.
    """

    def __init__(
        self,
        *,
        categories: Union[str, List[List[str]]] = "auto",
        descriptors: Union[str, List[List[str]]] = "auto",
        values: List[List[List[float]]],
        sparse: bool = False,
        dtype=np.float64,
        handle_unknown="error",
    ):
        """Encoder to translate categorical parameters into continuous descriptor values.

        Args:
            values (List[List[List[float]]]): Nested list of descriptor values. Must be of shape (n_features, n_categories_per_feature, n_descriptors).
            categories (List[List[str]], optional): List of strings referring to the categories (not the feature names!) occuring in the dataset.
                                                    Defaults to "auto". (When no list is passed, the descriptor names are generated automatically.)
            descriptors (List[List[str]], optional): List of strings referring to the descriptor names occuring in the dataset.
                                                    Defaults to "auto". (When no list is passed, the descriptor names are generated automatically.)
            sparse (bool, optional): Sparse matrix output is currently not supported. Defaults to False.
            dtype (dtype, optional): [description]. Defaults to np.float64.
            handle_unknown (str, optional): Allows to distinguish between "error" and "ignore". When "ignore", onknown categories are encoded as zeros. Defaults to "error".
        """
        self.categories = categories
        self.descriptors = descriptors
        self.values = values

        # check, whether we have only 1D data or multiple descriptors provided at once
        if not isinstance(self.values[0][0], list):
            self.values = [values]

        if not isinstance(self.categories[0], list) and self.categories != "auto":
            self.categories = [categories]

        if not isinstance(self.descriptors[0], list) and self.descriptors != "auto":
            self.descriptors = [descriptors]

        self.sparse = sparse
        self.dtype = dtype
        self.handle_unknown = handle_unknown

    def _validate_keywords(self):
        """Validate `handle_unknown` argument.

        Raises:
            ValueError: if `handle_unknown`not `error` or `ignore`.
        """
        if self.handle_unknown not in ("error", "ignore"):
            msg = (
                "handle_unknown should be either 'error' or 'ignore', got {0}.".format(
                    self.handle_unknown
                )
            )
            raise ValueError(msg)

    def _fit(
        self, X, handle_unknown="error", force_all_finite: Union[str, bool] = True
    ):
        self._check_n_features(X, reset=True)
        self._check_feature_names(X, reset=True)
        X_list, n_samples, n_features = self._check_X(
            X, force_all_finite=force_all_finite  # type: ignore
        )
        self.n_features_in_ = n_features

        if self.categories != "auto":
            if len(self.categories) != n_features:
                raise ValueError(
                    "Shape mismatch: if categories is an array,"
                    " it has to be of shape (n_features,)."
                )

        if self.descriptors != "auto":
            for i, des in enumerate(self.descriptors):
                if len(des) != len(self.values[i][0]):
                    raise ValueError(
                        "Shape mismatch: number of descriptors"
                        " do not fit to the dimension of provided values."
                    )

        self.values_ = []
        self.n_categories_i = []
        self.n_descriptors_in_ = []

        for values in self.values:
            descriptor_list, n_categories_i, n_descriptors = self._check_X(
                values, force_all_finite=force_all_finite  # type: ignore
            )
            self.values_.append(descriptor_list)
            self.n_categories_i.append(n_categories_i)
            self.n_descriptors_in_.append(n_descriptors)

        self.categories_ = []
        self.descriptors_ = []

        for i in range(n_features):
            Xi = X_list[i]
            if self.categories == "auto":
                cats = _unique(Xi)
            else:
                cats = np.array(self.categories[i], dtype=Xi.dtype)
                if Xi.dtype.kind not in "OUS":
                    sorted_cats = np.sort(cats)
                    error_msg = (
                        "Unsorted categories are not supported for numerical categories"
                    )
                    # if there are nans, nan should be the last element
                    stop_idx = -1 if np.isnan(sorted_cats[-1]) else None
                    if np.any(sorted_cats[:stop_idx] != cats[:stop_idx]) or (
                        np.isnan(sorted_cats[-1]) and not np.isnan(sorted_cats[-1])
                    ):
                        raise ValueError(error_msg)

                if handle_unknown == "error":
                    diff = _check_unknown(Xi, cats)
                    if diff:
                        msg = (
                            "Found unknown categories {0} in column {1}"
                            " during fit".format(diff, i)
                        )
                        raise ValueError(msg)
            self.categories_.append(cats)

            des_list = []
            for j in range(self.n_descriptors_in_[i]):
                if len(self.values_[i][j]) != len(cats):
                    raise ValueError(
                        "Shape mismatch: descriptor values has to be of shape (n_categories_per_feature, n_descriptors)."
                    )
                if self.descriptors == "auto":
                    des_list.append(f"Descriptor_{i}_{j}")

                else:
                    des_list.append(np.array(self.descriptors[i][j]))

            self.descriptors_.append(des_list)

    def fit(self, X, y=None):
        """
        Fit Encoder to X.

        Args:
            X (array-like of shape (n_samples, n_features)): The data to determine the categories of each feature.
            y: None. Ignored. This parameter exists only for compatibility with :class:`~sklearn.pipeline.Pipeline`.

        Returns:
            self: Fitted encoder.
        """
        self._validate_keywords()
        self._fit(X, handle_unknown=self.handle_unknown, force_all_finite="allow-nan")
        return self

    def fit_transform(self, X, y=None):
        """
        Fit categoricalDescriptorEncoder to X, then transform X.
        Equivalent to fit(X).transform(X) but more convenient.

        Args:
            X (array-like of shape (n_samples, n_features)): The data to encode.
            y: None. Ignored. This parameter exists only for compatibility with :class:`~sklearn.pipeline.Pipeline`.

        Returns:
            X_out ({ndarray, matrix} of shape (n_samples, n_encoded_features)): Transformed input.
        """
        self._validate_keywords()
        return super().fit_transform(X, y)

    def transform(self, X):
        """
        Transform X using descriptors.

        Args:
            X (array-like of shape (n_samples, n_features)): The data to encode.

        Returns:
            X_out ({ndarray, sparse matrix} of shape (n_samples, n_encoded_features)): Transformed input.
        """
        check_is_fitted(self)

        # validation of X happens in _check_X called by _transform
        warn_on_unknown = self.handle_unknown == "ignore"
        X_int, X_mask = self._transform(
            X,
            handle_unknown=self.handle_unknown,
            force_all_finite="allow-nan",  # type: ignore
            warn_on_unknown=warn_on_unknown,
        )

        n_samples = X.shape[0]
        n_values = [len(cats) for cats in self.categories_]

        X_tr = np.zeros((n_samples, sum(self.n_descriptors_in_)), dtype=float)

        for c, categories in enumerate(self.categories_):
            for i, category in enumerate(categories):
                for j in range(self.n_descriptors_in_[c]):
                    col = sum(self.n_descriptors_in_[:c])
                    # insert values to categorical data
                    row = np.where(X == category)[0]
                    X_tr[row, col + j] = self.values_[c][j][i]

        ### check whats going on here ### #TODO: sparse matrix is currently not supported!
        mask = X_mask.ravel()
        feature_indices = np.cumsum([0] + n_values)
        indices = (X_int + feature_indices[:-1]).ravel()[mask]

        indptr = np.empty(n_samples + 1, dtype=int)
        indptr[0] = 0
        np.sum(X_mask, axis=1, out=indptr[1:])
        np.cumsum(indptr[1:], out=indptr[1:])
        data = np.ones(indptr[-1])

        _ = sparse.csr_matrix(
            (data, indices, indptr),
            shape=(n_samples, feature_indices[-1]),
            dtype=self.dtype,
        )
        ####################################
        if not self.sparse:
            return X_tr
        else:
            raise NotImplementedError(
                "Sparse matrices as output are not implemented yet"
            )
            # return X_tr_sparse

    def inverse_transform(self, X):
        """
        Convert the data back to the original representation.

        Args:
            X ({array-like, sparse matrix} of shape (n_samples, n_encoded_features*n_descriptors_per_feature)): The transformed data.

        Returns:
            X_tr (ndarray of shape (n_samples, n_features)): Inverse transformed array.
        """
        check_is_fitted(self)
        X = check_array(X, accept_sparse="csr")
        n_samples = X.shape[0]

        # validate shape of passed X
        msg = (
            "Shape of the passed X data is not correct. Expected {0} columns, got {1}."
        )
        if X.shape[1] != sum(self.n_descriptors_in_):
            raise ValueError(msg.format(len(self.descriptors), X.shape[1]))

        X_tr = np.empty((n_samples, self.n_features_in_), dtype=object)

        for c, categories in enumerate(self.categories_):
            indices = np.cumsum([0] + self.n_descriptors_in_)

            var_descriptor_conditions = X[:, indices[c] : indices[c + 1]]
            var_descriptor_orig_data = np.column_stack(self.values_[c])
            var_categorical_transformed = []
            # Find the closest points by euclidean distance
            for i in range(n_samples):
                # Euclidean distance calculation
                eucl_distance_squ = np.sum(
                    np.square(
                        np.subtract(
                            var_descriptor_orig_data,
                            var_descriptor_conditions[i, :],
                        )
                    ),
                    axis=1,
                )
                # Choose closest point
                category_index = np.where(
                    eucl_distance_squ == np.min(eucl_distance_squ)
                )[0][0]
                # Find the matching name of the categorical variable
                category = categories[category_index]
                # Add the original categorical variables name to the dataset
                var_categorical_transformed.append(category)
            X_tr[:, c] = np.asarray(var_categorical_transformed)

        return X_tr

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation.

        Args:
            input_features (array-like of str or None): default=None.
                Input features.
                - If `input_features` is `None`, then `feature_names_in_` is
                    used as feature names in. If `feature_names_in_` is not defined,
                    then names are generated: `[x0, x1, ..., x(n_features_in_)]`.
                - If `input_features` is an array-like, then `input_features` must
                    match `feature_names_in_` if `feature_names_in_` is defined.

        Returns:
            feature_names_out (ndarray(str)): Transformed feature names.
        """
        check_is_fitted(self)

        desc = self.descriptors_
        input_features = _check_feature_names_in(self, input_features)

        feature_names = []
        for i in range(len(desc)):
            names = [input_features[i] + "_" + str(t) for t in desc[i]]
            feature_names.extend(names)
        return np.asarray(feature_names, dtype=object)
