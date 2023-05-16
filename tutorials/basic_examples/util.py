import numpy as np
import pandas as pd

# Reaction Optimization Notebook util code

T0 = 25
T1 = 100
e0 = np.exp((T1 + 0) / T0)
e60 = np.exp((T1 + 60) / T0)
de = e60 - e0

boiling_points = {  # in °C
    "MeOH": 64.7,
    "THF": 66.0,
    "Dioxane": 101.0,
}
density = {  # in kg/l
    "MeOH": 0.792,
    "THF": 0.886,
    "Dioxane": 1.03,
}
# create dict from individual dicts
descs = {
    "boiling_points": boiling_points,
    "density": density,
}
solvent_descriptors = pd.DataFrame(descs)


# these functions are for faking real experimental data ;)
def calc_volume_fact(V):
    # 20-90
    # max at 75 = 1
    # min at 20 = 0.7
    # at 90=0.5
    x = (V - 20) / 70
    x = 0.5 + (x - 0.75) * 0.1 + (x - 0.4) ** 2
    return x


def calc_rhofact(solvent_type, Tfact):
    #  between 0.7 and 1.1
    x = solvent_descriptors["density"][solvent_type]
    x = (1.5 - x) * (Tfact + 0.5) / 2
    return x.values


def calc_Tfact(T):
    x = np.exp((T1 + T) / T0)
    return (x - e0) / de


# this can be used to create a dataframe of experiments including yields
def create_experiments(domain, nsamples=100, A=25, B=90, candidates=None):
    Tf = domain.inputs.get_by_key("Temperature")
    Vf = domain.inputs.get_by_key("Solvent Volume")
    typef = domain.inputs.get_by_key("Solvent Type")
    yf = domain.outputs.get_by_key("Yield")
    if candidates is None:
        T = np.random.uniform(low=Tf.lower_bound, high=Tf.upper_bound, size=(nsamples,))
        V = np.random.uniform(low=Vf.lower_bound, high=Vf.upper_bound, size=(nsamples,))
        solvent_types = [
            domain.inputs.get_by_key("Solvent Type").categories[np.random.randint(0, 3)]
            for i in range(nsamples)
        ]
    else:
        nsamples = len(candidates)
        T = candidates["Temperature"].values
        V = candidates["Solvent Volume"].values
        solvent_types = candidates["Solvent Type"].values

    Tfact = calc_Tfact(T)
    rhofact = calc_rhofact(solvent_types, Tfact)
    Vfact = calc_volume_fact(V)
    y = A * Tfact + B * rhofact
    y = 0.5 * y + 0.5 * y * Vfact
    # y = y.values
    samples = pd.DataFrame(
        {
            Tf.key: T,
            Vf.key: V,
            yf.key: y,
            typef.key: solvent_types,
            "valid_" + yf.key: np.ones(nsamples),
        },
        # index=pd.RangeIndex(nsamples),
    )
    samples.index = pd.RangeIndex(nsamples)
    return samples


def create_candidates(domain, nsamples=4):
    experiments = create_experiments(domain, nsamples=nsamples)
    candidates = experiments.drop(["Yield", "valid_Yield"], axis=1)
    return candidates


# this is for evaluating candidates that do not yet have a yield attributed to it.
def evaluate_experiments(domain, candidates):
    return create_experiments(domain, candidates=candidates)


evaluate_candidates = evaluate_experiments
