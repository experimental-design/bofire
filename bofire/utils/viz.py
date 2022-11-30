import itertools
import math
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from bofire.domain import Domain
from bofire.domain.features import ContinuousOutput, OutputFeature
from bofire.domain.objectives import IdentityObjective
from bofire.utils.multiobjective import get_pareto_front


def plot_pareto_fronts(
    domain: Domain,
    experiments: pd.DataFrame,
    output_feature_keys: Optional[list] = None,
    ncols: int = 3,
):
    if output_feature_keys is None:
        output_feature_keys = domain.output_features.get_by_objective(excludes=None)
    else:
        assert (
            len(output_feature_keys) >= 2
        ), "At least two output feature keys has to be provided."
        for key in output_feature_keys:
            feat = domain.get_feature(key)
            assert isinstance(feat, ContinuousOutput)
            assert isinstance(feat.objective, IdentityObjective)

    assert len(output_feature_keys) > 1, "Only one output feature in domain."
    combis = list(itertools.combinations(output_feature_keys, 2))
    nrows = math.ceil(len(combis) / ncols)
    ncols = ncols if len(combis) >= ncols else len(combis)

    experiments = domain.preprocess_experiments_all_valid_outputs(
        experiments, output_feature_keys
    )
    pareto_experiments = get_pareto_front(domain, experiments, output_feature_keys)

    fig, axes = plt.subplots(
        ncols=ncols, nrows=nrows, figsize=[6.4 * ncols, 4.8 * nrows]
    )

    for i, combi in enumerate(combis):
        if len(combis) == 1:
            ax = axes
        elif len(combis) <= ncols:
            ax = axes[i]
        else:
            irow = i // ncols
            icol = i % ncols
            ax = axes[irow, icol]
        specific_pareto_experiments = get_pareto_front(domain, experiments, list(combi))
        ax.scatter(experiments[combi[0]], experiments[combi[1]], label="all")
        ax.scatter(
            pareto_experiments[combi[0]], pareto_experiments[combi[1]], label="optimal"
        )
        ax.scatter(
            specific_pareto_experiments[combi[0]],
            specific_pareto_experiments[combi[1]],
            label="specific",
        )
        ax.set_xlabel(combi[0])
        ax.set_ylabel(combi[1])
        ax.legend()
    # set blanks
    if (nrows * ncols > len(output_feature_keys)) and (nrows > 1):
        for i in range((nrows * ncols) % len(output_feature_keys)):
            axes[-1, -1 * (i + 1)].axis("off")
    return fig, axes


def plot_hists(keys, experiments: pd.DataFrame, ncols: int = 3):
    nrows = math.ceil(len(keys) / ncols)
    ncols = ncols if len(keys) >= ncols else len(keys)
    axes = experiments.hist(
        keys, figsize=[4.8 * ncols, 4 * nrows], layout=(nrows, ncols)
    )
    return axes


def plot_duplicates(
    experiments: pd.DataFrame,
    duplicated_labcodes: list,
    output_features: list,
    ncols: int = 3,
):
    nrows = math.ceil(len(output_features) / ncols)
    ncols = ncols if len(output_features) >= ncols else len(output_features)
    fig, axes = plt.subplots(
        ncols=ncols, nrows=nrows, figsize=[6.4 * ncols, 4.8 * nrows]
    )
    for j, output_feature in enumerate(output_features):
        if len(output_features) == 1:
            ax = axes
        elif len(output_features) <= ncols:
            ax = axes[j]
        else:
            irow = j // ncols
            icol = j % ncols
            ax = axes[irow, icol]
        for i, d in enumerate(duplicated_labcodes):
            # plot all
            data = experiments.loc[
                experiments.labcode.isin(d) & experiments[output_feature].notna(),
                output_feature,
            ].values
            ax.scatter(len(data) * [i], data)
            # mark invalids
            data = experiments.loc[
                experiments.labcode.isin(d)
                & experiments[output_feature].notna()
                & (experiments[f"valid_{output_feature}"] == 0),
                output_feature,
            ].values
            ax.scatter(len(data) * [i], data, color="black", marker="x")
        ax.set_xticks(list(range(len(duplicated_labcodes))))
        ax.set_xticklabels(
            [lcs[0] for lcs in duplicated_labcodes], rotation=45.0, ha="right"
        )
        ax.set_ylabel(output_feature)
        ax.set_title(f"Duplicates {output_feature}")
        ax.grid(axis="x")
        plt.tight_layout()
    # set blanks
    if (nrows * ncols > len(output_features)) and (nrows > 1):
        for i in range((nrows * ncols) % len(output_features)):
            axes[-1, -1 * (i + 1)].axis("off")

    return fig, ax


def parity(
    datasets: List[Tuple[np.array, np.array]],
    datalabels: Optional[List[str]] = None,
    uncertainties: Optional[List[np.array]] = None,
    uncertainty_scale: int = 1,
    hoffset: Optional[float] = None,
    title: str = "",
    ax: Optional[Axes] = None,
):
    if datalabels is None:
        datalabels = [None for i in range(len(datasets))]
    assert len(datasets) == len(datalabels)
    if uncertainties is None:
        uncertainties = [None for i in range(len(datalabels))]
    assert len(datasets) == len(uncertainties)

    if ax is None:
        _, ax = plt.subplots()

    ymin = np.inf
    ymax = -np.inf
    for i, dset in enumerate(datasets):
        ax.scatter(dset[0], dset[1], label=datalabels[i])
        if uncertainties[i] is not None:
            ax.errorbar(
                dset[0], dset[1], yerr=uncertainties[i] * uncertainty_scale, fmt="none"
            )
        if dset[0].max() > ymax:
            ymax = dset[0].max()
        if dset[0].min() < ymin:
            ymin = dset[0].min()
    ax.plot([ymin, ymax], [ymin, ymax], color="black")
    if hoffset is not None:
        ax.plot([ymin, ymax], [ymin - hoffset, ymax - hoffset], color="black", ls="--")
        ax.plot([ymin, ymax], [ymin + hoffset, ymax + hoffset], color="black", ls="--")
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    ax.legend()
    return ax


def plot_fi(
    featurenames,
    importances,
    std=None,
    significance=None,
    comment: str = "",
    sort: bool = True,
):
    """[summary]

    Args:
        featurenames (List[str]): List of feature names
        importances (np.ndarray): The importances that should be plotted
        std (np.ndarray, optional): The uncertainties of the importances. Defaults to None.
        comment (str, optional): A comment mentioned in the title of the plot. Defaults to "".
    """
    if sort:
        indices = np.argsort(importances)
    else:
        indices = np.array(list(range(len(importances))))
    color = ["r" for i in range(len(importances))]
    if significance is not None:
        color = np.array(color)
        color[modelres.pvalues.values[1:] < 0.05] = "g"
        color = color.tolist()
    plt.figure()
    plt.title("Feature importance %s" % comment)
    plt.barh(
        range(importances.shape[0]),
        importances[indices],
        color=color,
        align="center",
        xerr=None if std is None else std[indices],
    )
    # If you want to define your own labels,
    # change indices to a list of labels on the following line.
    plt.yticks(range(importances.shape[0]), [featurenames[i] for i in indices])
    # plt.ylim([-1, X.shape[1]])
    plt.show()
    return
