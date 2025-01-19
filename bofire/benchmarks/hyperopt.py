from typing import Optional

import pandas as pd
from tqdm import tqdm

import bofire.surrogates.api as surrogates
from bofire.benchmarks.benchmark import Benchmark
from bofire.data_models.domain.api import Domain
from bofire.data_models.surrogates.api import AnyTrainableSurrogate


class Hyperopt(Benchmark):
    def __init__(
        self,
        surrogate_data: AnyTrainableSurrogate,
        training_data: pd.DataFrame,
        folds: int,
        random_state: Optional[int] = None,
        show_progress_bar: bool = False,
    ) -> None:
        super().__init__()
        if surrogate_data.hyperconfig is None:
            raise ValueError("No hyperoptimization configuration found.")
        self.surrogate_data = surrogate_data
        self.training_data = training_data
        self.folds = folds
        self.results = None
        self.random_state = random_state
        self.show_progress_bar = show_progress_bar

    @property
    def domain(self) -> Domain:
        return self.surrogate_data.hyperconfig.domain  # type: ignore

    @property
    def target_metric(self):
        return self.surrogate_data.hyperconfig.target_metric  # type: ignore

    def _f(self, candidates: pd.DataFrame) -> pd.DataFrame:
        for i, candidate in tqdm(
            candidates.iterrows(),
            total=candidates.shape[0],
            disable=self.show_progress_bar is False,
        ):
            self.surrogate_data.update_hyperparameters(candidate)
            surrogate = surrogates.map(self.surrogate_data)
            _, cv_test, _ = surrogate.cross_validate(  # type: ignore
                self.training_data,
                folds=self.folds,
                random_state=self.random_state,
            )
            if i == 0:
                results = cv_test.get_metrics(combine_folds=True)
            else:
                results = pd.concat(
                    [results, cv_test.get_metrics(combine_folds=True)],  # type: ignore
                    ignore_index=True,
                    axis=0,
                )
        results[f"valid_{self.target_metric.value}"] = 1  # type: ignore
        return results  # type: ignore
