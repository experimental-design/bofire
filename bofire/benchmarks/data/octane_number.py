# Dataset consisting of motor octane numbers of hydrocarbon mixtures
# taken from https://www.nature.com/articles/s41524-025-01552-2
# and formatted to the needs of BoFire
import pandas as pd


def get_octane_data() -> pd.DataFrame:
    df_experiments = pd.read_csv("octane.csv")
    return df_experiments
