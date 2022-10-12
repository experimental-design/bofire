from typing import List, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from everest.study import PoolStudy, Study
from matplotlib.axes import Axes


# TODO: Maybe unite both plots
def plot_optimization_history(
    study: Union[Study, Sequence[Study]],
    error_bar: bool = False,
    log_scale: bool = False,
    ax: Optional[Axes] = None,
    best_possible_f: Optional[float] = None,
    label: str = '',
    x_axis: str = "iteration"
):
    assert x_axis in ["iteration", "evaluation"]
    
    if isinstance(study, Study):
        studies = [study]
    else:
        studies = list(study)

    if best_possible_f is None:
        best_possible_f = studies[0].best_possible_f

    for study in studies:
        if log_scale:
            if best_possible_f is None:
                ylabel = "Log(fbest)"
                study.meta["viz"] = np.log10(study.meta.fbest)
            else:
                ylabel = "Log(Difference to best)"
                study.meta["viz"] = np.log10(np.abs(best_possible_f - study.meta.fbest))
        else:
            if best_possible_f is None:
                ylabel = "fbest"
                study.meta["viz"] = study.meta.fbest
            else:
                ylabel = "Difference to best"
                study.meta["viz"] = best_possible_f - study.meta.fbest

    if x_axis == "iteration":
        fbests = [study.meta.groupby("batch", as_index=False)["viz"].mean() for study in studies]
    else:
        fbests = [study.meta for study in studies]
    
    if ax is None:
        _, ax = plt.subplots()

    if error_bar:

        means = np.mean(np.array([i["viz"].values.tolist() for i in fbests]),axis=0)
        stds = np.std(np.array([i["viz"].values.tolist() for i in fbests]),axis=0)

        ax.plot(means, label = label)
        ax.fill_between(range(means.shape[0]), means - stds, means + stds, alpha = 0.2)
        
    else:
        for i, study in enumerate(studies):
            ax.plot(fbests[i]["viz"], marker = ".", label = label+str(i+1))
    
    if x_axis == "iteration":
        ax.set_xlabel("Iteration")
    else:
        ax.set_xlabel("Function Evaluations")
    ax.set_ylabel(ylabel)
    return ax


def plot_pooloptimization_history(
    study: Union[PoolStudy, Sequence[PoolStudy]],
    error_bar: bool = False,
    log_scale: bool = False,
    ax: Optional[Axes] = None,
    label: str = '',
):
    if isinstance(study, PoolStudy):
        studies = [study]
    else:
        studies = list(study)

    trajecs = []
    
    for study in studies:
        # generate trajectory
        trajec = np.array([study.get_fbest(study.experiments.loc[study.meta.iteration.notna() & (study.meta.iteration<=i)]) for i in range(study.num_iterations)])
        if log_scale:
            trajecs.append(np.log10(np.abs(study.get_fbest(study.experiments)- trajec)))
            ylabel = "Log(Difference to best)"
        else:
            ylabel = "Difference to best"
            trajecs.append(study.get_fbest(study.experiments)- trajec)

    if ax is None:
        _, ax = plt.subplots()

    if error_bar:

        means = np.mean(np.array([trajecs[i].tolist() for i in range(len(trajecs))]),axis=0)
        stds = np.std(np.array([trajecs[i].tolist() for i in range(len(trajecs))]),axis=0)

        ax.plot(means, label = label)
        ax.fill_between(range(means.shape[0]), means - stds, means + stds, alpha = 0.2)
        
    else:
        for i, trajec in enumerate(trajecs):
            ax.plot(trajec, marker = ".", label = label+str(i+1))

    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    return ax
