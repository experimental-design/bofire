from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from scipy.stats import chi2

import bofire.surrogates.api as surrogates
from bofire.data_models.domain.api import Inputs, Outputs


class OutlierDetection(ABC):
    @abstractmethod
    def detect(self, experiments: pd.DataFrame) -> pd.DataFrame:
        pass

    @property
    @abstractmethod
    def inputs(self) -> Inputs:
        pass

    @property
    @abstractmethod
    def outputs(self) -> Outputs:
        pass


class IterativeTrimming(OutlierDetection):
    def __init__(self, data_model, **kwargs):
        self.alpha1 = data_model.alpha1
        self.alpha2 = data_model.alpha2
        self.nsh = data_model.nsh
        self.ncc = data_model.ncc
        self.nrw = data_model.nrw
        self.base_gp = data_model.base_gp
        self.surrogate = surrogates.map(self.base_gp)
        super().__init__()

    @property
    def inputs(self) -> Inputs:
        return self.base_gp.inputs

    @property
    def outputs(self) -> Outputs:
        return self.base_gp.outputs

    def detect(self, experiments: pd.DataFrame) -> pd.DataFrame:
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
                ix_sub = (
                    d_sq <= np.partition(d_sq, h)[h]  # type: ignore
                )  # alpha-quantile
                consistency = alpha / chi2(p + 2).cdf(chi_sq)

            # check convergence
            if (i > self.nsh + 1) and (ix_sub == ix_old).all():  # type: ignore
                break  # converged
            ix_old = ix_sub

            self.surrogate.fit(  # type: ignore
                experiments[experiments.index.isin(indices[ix_sub])].copy(),
            )
            # make prediction
            pred = self.surrogate.predict(experiments)
            d_sq = (
                (
                    (
                        experiments[self.base_gp.outputs.get_keys()[0]]
                        - pred[self.base_gp.outputs.get_keys()[0] + "_pred"]
                    )
                    ** 2
                    / pred[self.base_gp.outputs.get_keys()[0] + "_sd"] ** 2
                )
                .to_numpy()
                .ravel()
            )

            niter += 1
        for _ in range(self.nrw):
            alpha = self.alpha2
            chi_sq = chi2(p).ppf(alpha)

            # XXX: might be buggy when there are identical data points
            ix_sub = d_sq <= chi_sq * consistency  # type: ignore
            consistency = alpha / chi2(p + 2).cdf(chi_sq)

            # check convergence
            if (ix_sub == ix_old).all():
                break  # converged
            ix_old = ix_sub

        filtered_experiments = experiments.copy()
        output_name = self.base_gp.outputs.get_keys()[0]
        filtered_experiments[f"valid_{output_name}"] = filtered_experiments[
            f"valid_{output_name}"
        ].astype(int)
        filtered_experiments.loc[
            ~ix_sub,  # type: ignore
            f"valid_{output_name}",
        ] = 0
        return filtered_experiments
