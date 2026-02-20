"""
SVM benchmark.

Adapted from https://github.com/LeoIV/BenchSuite/blob/master/benchsuite/svm.py
"""

import gzip
import logging
import urllib
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

from bofire.benchmarks.benchmark import Benchmark
from bofire.data_models.domain.api import Domain, Inputs, Outputs
from bofire.data_models.features.api import ContinuousInput, ContinuousOutput
from bofire.data_models.objectives.api import MinimizeObjective


class SVM(Benchmark):
    """
    SVM benchmark task.

    This benchmark evaluates the performance of Support Vector Regression (SVR)
    with different hyperparameter configurations on a CT slice dataset.
    """

    def __init__(self, **kwargs):
        """Initialize the SVM benchmark.

        Args:
            **kwargs: Additional arguments for the Benchmark class.
        """
        super().__init__(**kwargs)

        self.dim = 388
        """
        seed from url=("https://github.com/hvarfner/vanilla_bo_in_highdim/blob/
        8174f6322d12154b4f84448a0bb54b71e56ffede/BenchSuite/benchsuite/svm.py#L30")
        """
        self.seed = 388
        np.random.seed(self.seed)

        # Define domain with continuous inputs for all hyperparameters
        input_features = [
            ContinuousInput(key=f"x_{i + 1}", bounds=[0.0, 1.0])
            for i in range(self.dim)
        ]

        output_features = [ContinuousOutput(key="y", objective=MinimizeObjective())]

        self._domain = Domain(
            inputs=Inputs(features=input_features),
            outputs=Outputs(features=output_features),
        )

        # Load SVM data
        X_np, y_np = self.get_data()
        X = MinMaxScaler().fit_transform(X_np)
        y = MinMaxScaler().fit_transform(y_np[:, None]).squeeze(-1)

        # Make train/test split
        idxs = np.random.choice(np.arange(len(X)), min(500, len(X)), replace=False)
        half = len(idxs) // 2
        self._X_train = X[idxs[:half]]
        self._X_test = X[idxs[half:]]
        self._y_train = y[idxs[:half]]
        self._y_test = y[idxs[half:]]

    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        url_X = (
            "https://github.com/LeoIV/BenchSuite/raw/"
            "73de8c581aacf2dc99120d9cf65b79cbfe2aaf4e/data/svm/CT_slice_X.npy.gz"
        )
        url_y = (
            "https://github.com/LeoIV/BenchSuite/raw/"
            "73de8c581aacf2dc99120d9cf65b79cbfe2aaf4e/data/svm/CT_slice_y.npy.gz"
        )
        print("Downloading SVM data...")
        try:
            with urllib.request.urlopen(url_X) as response:
                with gzip.GzipFile(fileobj=response) as f:
                    X_np = np.load(f)
            with urllib.request.urlopen(url_y) as response:
                with gzip.GzipFile(fileobj=response) as f:
                    y_np = np.load(f)
            print("Download complete.")
            return X_np, y_np
        except Exception as e:
            logging.error("Error downloading or loading data: %s", e)
            raise e

    def _evaluate_single(self, hypers: np.ndarray) -> float:
        """
        Evaluate SVM error for one set of hyperparameters.

        Args:
            hypers: One input hyperparameter configuration (388-dimensional).

        Returns:
            SVM prediction error (RMSE).
        """
        C = 0.01 * (500 ** hypers[387])
        gamma = 0.1 * (30 ** hypers[386])
        epsilon = 0.01 * (100 ** hypers[385])
        lengthscales = np.exp(4 * hypers[:385] - 2)

        svr = SVR(gamma=gamma, epsilon=epsilon, C=C, cache_size=1500, tol=0.001)
        svr.fit(self._X_train / lengthscales, self._y_train)
        pred = svr.predict(self._X_test / lengthscales)
        error = np.sqrt(np.mean(np.square(pred - self._y_test)))
        return error

    def _f(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluate SVM benchmark for a batch of candidates.

        Args:
            candidates: DataFrame with columns x_1, x_2, ..., x_388 containing
                       hyperparameter configurations in [0, 1].

        Returns:
            DataFrame with columns 'y' (SVM error) and 'valid_y' (always 1).
        """
        # Extract hyperparameters from candidates DataFrame
        hypers_columns = [f"x_{i + 1}" for i in range(self.dim)]
        hypers = candidates[hypers_columns].values

        # Evaluate SVM for each set of hyperparameters
        errors = np.array([self._evaluate_single(h) for h in hypers])

        # Return as DataFrame
        return pd.DataFrame({"y": errors, "valid_y": 1})

    def get_optima(self) -> pd.DataFrame:
        """Get the optima for the SVM benchmark.

        Returns:
            DataFrame with optimal hyperparameters and corresponding error.
            For this benchmark, the true optimum is not known, so this raises
            NotImplementedError.
        """
        raise NotImplementedError(
            "The true optimum for the SVM benchmark is not known."
        )
