from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2

from bofire.surrogates.api import SingleTaskGPSurrogate


class OutlierDetection(ABC):
    @abstractmethod
    def detect(self, experiments: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        pass


class IterativeTrimming(OutlierDetection):
    def __init__(self, data_model, **kwargs):
        self.alpha1 = data_model.alpha1
        self.alpha2 = data_model.alpha2
        self.nsh = data_model.nsh
        self.ncc = data_model.ncc
        self.nrw = data_model.nrw
        self.base_gp = data_model.base_gp
        super().__init__()

    def detect(self, experiments: pd.DataFrame):
        model = SingleTaskGPSurrogate(data_model=self.base_gp)
        n = len(experiments)
        indices = experiments.index.to_numpy()
        p = 1
        if n * self.alpha1 - 0.5 <= 2:
            raise ValueError("The dataset is unreasonably small!")
        d_sq = None
        ix_old = None
        niter = 0
        for i in range(1 + self.nsh + self.ncc):
            if i == 0:
                # starting with the full sample
                ix_sub = slice(None)
                consistency = 1.0
            else:
                # reducing alpha from 1 to alpha1 gradually
                if i <= self.nsh:
                    alpha = self.alpha1 + (1 - self.alpha1) * (1 - i / (self.nsh + 1))
                else:
                    alpha = self.alpha1
                chi_sq = chi2(p).ppf(alpha)
                h = int(min(np.ceil(n * alpha - 0.5), n - 1))  # alpha <= (h+0.5)/n

                # XXX: might be buggy when there are identical data points
                # better to use argpartition! but may break ix_sub == ix_old.
                ix_sub = d_sq <= np.partition(d_sq, h)[h]  # alpha-quantile
                consistency = alpha / chi2(p + 2).cdf(chi_sq)

            # check convergence
            if (i > self.nsh + 1) and (ix_sub == ix_old).all():
                break  # converged
            ix_old = ix_sub

            model.fit(experiments[experiments.index.isin(indices[ix_sub])])

            # make prediction
            pred = model.predict(experiments)
            d_sq = (
                (experiments["y"] - pred["y_pred"]) ** 2 / pred["y_sd"] ** 2
            ).ravel()

            niter += 1
        for _ in range(self.nrw):
            alpha = self.alpha2
            chi_sq = chi2(p).ppf(alpha)

            # XXX: might be buggy when there are identical data points
            ix_sub = d_sq <= chi_sq * consistency
            consistency = alpha / chi2(p + 2).cdf(chi_sq)

            # check convergence
            if (ix_sub == ix_old).all():
                break  # converged
            ix_old = ix_sub
        return (
            experiments[experiments.index.isin(indices[ix_sub])],
            experiments[~experiments.index.isin(indices[ix_sub])],
        )
