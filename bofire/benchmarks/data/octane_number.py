# Dataset consisting of motor octane numbers of hydrocarbon mixtures
# taken from https://www.nature.com/articles/s41524-025-01552-2
# and formatted to the needs of BoFire
from importlib import resources

import pandas as pd


def get_octane_data() -> pd.DataFrame:
    data_path = resources.files("bofire.benchmarks.data").joinpath("octane.csv")
    return pd.read_csv(data_path)
