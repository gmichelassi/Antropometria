import math
import numpy as np
import pandas as pd


def apply_min_max_normalization(df: pd.DataFrame) -> pd.DataFrame:
    df_final = []

    max_dist = math.ceil(np.amax(df.to_numpy()))
    min_dist = math.floor(np.amin(df.to_numpy()))

    for feature, data in df.iteritems():
        columns = []
        for i in data.values:
            xi = (i - min_dist)/(max_dist - min_dist)
            columns.append(xi)
        df_final.append(columns)

    return pd.DataFrame(df_final, dtype=float).T
